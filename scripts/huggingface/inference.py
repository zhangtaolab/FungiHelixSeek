#!/usr/bin/env python3
"""MLM mask-recovery inference for FungiHelixSeek.

Reads a JSONL file whose lines contain at least ``masked_sequence`` (with
``<mask>`` placeholders).  Optional fields: ``original_sequence``,
``mask_positions``, ``masked_bases``, ``id``.

Writes an output JSONL with per-mask top-k predictions and overall accuracy::

    python scripts/huggingface/inference.py \
        --model_path models/FungiHelixSeek \
        --input_file data/test_data/inference_example.jsonl \
        --output_file tmp/mask_recovery.jsonl \
        --top_k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from common import (
    DNA_BASES,
    get_device,
    get_dna_base_token_ids,
    load_model_and_tokenizer,
    normalize_mask_tokens,
    parse_torch_dtype,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FungiHelixSeek MLM mask recovery")
    p.add_argument("--model_path", "-m", default="models/FungiHelixSeek")
    p.add_argument("--input_file", "-i", required=True)
    p.add_argument("--output_file", "-o", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "mps", "cpu"])
    p.add_argument("--torch_dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


@torch.no_grad()
def recover_masks_batch(
    model, tokenizer, sequences, device, base_ids, top_k, max_length,
):
    results = []
    mask_token_id = tokenizer.mask_token_id

    for seq in sequences:
        normalized = normalize_mask_tokens(seq, tokenizer.mask_token)
        enc = tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]

        mask_preds = []
        for pos in mask_positions.tolist():
            row = logits[0, pos].float().softmax(dim=-1)

            valid_ids = list(base_ids.values())
            sub = row[valid_ids]
            k = min(top_k, len(valid_ids))
            vals, idx_in_sub = torch.topk(sub, k)
            global_ids = [valid_ids[i] for i in idx_in_sub.tolist()]
            tokens = tokenizer.convert_ids_to_tokens(global_ids)

            mask_preds.append({
                "position": pos,
                "top_k": [
                    {"base": t, "probability": round(v, 6)}
                    for t, v in zip(tokens, vals.tolist())
                ],
                "predicted_base": tokens[0],
            })

        predicted_bases = [mp["predicted_base"] for mp in mask_preds]
        reconstructed = list(normalized)
        ptr = 0
        for i, ch in enumerate(reconstructed):
            if ch == tokenizer.mask_token[0] and ptr < len(predicted_bases):
                mt = tokenizer.mask_token
                mt_len = len(mt)
                if "".join(reconstructed[i:i + mt_len]) == mt:
                    for j in range(mt_len):
                        reconstructed[i + j] = ""
                    reconstructed.insert(i, predicted_bases[ptr])
                    ptr += 1
        reconstructed_seq = "".join(reconstructed)

        results.append({
            "mask_predictions": mask_preds,
            "predicted_sequence": reconstructed_seq,
        })

    return results


def main():
    args = parse_args()
    device = get_device(args.device)
    dtype = parse_torch_dtype(args.torch_dtype)
    logger.info("Device: %s | dtype: %s", device, dtype)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, task_type="mlm", torch_dtype=dtype,
    )
    model.to(device).eval()

    base_ids = get_dna_base_token_ids(tokenizer)
    logger.info("DNA base token IDs: %s", base_ids)

    records = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), args.input_file)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct, total = 0, 0
    with open(out_path, "w") as fout:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            seqs = [r["masked_sequence"] for r in batch]
            preds = recover_masks_batch(
                model, tokenizer, seqs, device, base_ids, args.top_k, args.max_length,
            )
            for rec, pred in zip(batch, preds):
                out = {k: v for k, v in rec.items()}
                out.update(pred)

                if "masked_bases" in rec:
                    truth = rec["masked_bases"]
                    predicted = [mp["predicted_base"] for mp in pred["mask_predictions"]]
                    matches = sum(t == p for t, p in zip(truth, predicted))
                    correct += matches
                    total += len(truth)
                    out["accuracy"] = round(matches / max(len(truth), 1), 4)

                fout.write(json.dumps(out) + "\n")

    if total > 0:
        logger.info("Top-1 accuracy: %d/%d = %.2f%%", correct, total, 100 * correct / total)
    logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
