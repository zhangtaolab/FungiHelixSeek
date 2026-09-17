#!/usr/bin/env python3
"""Batch prediction with a fine-tuned FungiHelixSeek classifier.

Reads a CSV with a ``sequence`` column, writes a CSV with predictions::

    python scripts/huggingface/predict.py \
        --model_path tmp/finetune_output \
        --input_file data/test_data/predict_example.csv \
        --output_file tmp/predictions.csv \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd
import torch

from common import copy_remote_code_files, get_device, load_model, load_tokenizer, parse_torch_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FungiHelixSeek prediction")
    p.add_argument("--model_path", "-m", required=True, help="Fine-tuned model directory")
    p.add_argument("--input_file", "-i", required=True, help="CSV with 'sequence' column")
    p.add_argument("--output_file", "-o", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--torch_dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = parse_torch_dtype(args.torch_dtype)

    copy_remote_code_files(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, task_type="sequence_classification", torch_dtype=dtype)
    model.to(device).eval()

    num_labels = getattr(model.config, "num_labels", 2)
    is_regression = num_labels == 1
    id2label = getattr(model.config, "id2label", {i: str(i) for i in range(num_labels)})

    df = pd.read_csv(args.input_file)
    if "sequence" not in df.columns:
        raise ValueError(f"Input CSV must have a 'sequence' column. Found: {list(df.columns)}")
    sequences = [s.upper() for s in df["sequence"].tolist()]
    logger.info("Loaded %d sequences from %s", len(sequences), args.input_file)

    all_predictions = []
    all_probs = []

    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start:start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=args.max_length, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits

        if is_regression:
            vals = logits.squeeze(-1).cpu().tolist()
            all_predictions.extend(vals)
            all_probs.extend([None] * len(vals))
        else:
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            preds = probs.argmax(dim=-1).tolist()
            all_predictions.extend(preds)
            all_probs.extend(probs.tolist())

    out_df = df.copy()
    if is_regression:
        out_df["predicted_value"] = all_predictions
    else:
        out_df["predicted_label"] = [id2label.get(p, str(p)) for p in all_predictions]
        out_df["predicted_class"] = all_predictions
        for c in range(num_labels):
            label_name = id2label.get(c, str(c))
            out_df[f"probability_{label_name}"] = [pr[c] for pr in all_probs]

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", out_path, len(out_df))

    if not is_regression:
        from collections import Counter
        dist = Counter(all_predictions)
        logger.info("Label distribution: %s", dict(dist))


if __name__ == "__main__":
    main()
