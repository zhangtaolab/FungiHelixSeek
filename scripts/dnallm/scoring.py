#!/usr/bin/env python3
"""Score DNA sequences with FungiHelixSeek using DNALLM.

Computes per-sequence scores via embedding similarity, logits, probability,
or loss. Useful for variant effect prediction and sequence fitness scoring.

Requires: DNALLM >= 0.5.2 (pip install dnallm)

    python scripts/dnallm/scoring.py \
        --model_path models/FungiHelixSeek \
        --input_file data/test_data/predict_example.csv \
        --output_file tmp/scores.csv \
        --score_type loss
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek sequence scoring via DNALLM")
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--input_file", "-i", required=True,
                   help="CSV with 'sequence' column or list of sequences")
    p.add_argument("--output_file", "-o", required=True)
    p.add_argument("--score_type", default="loss",
                   choices=["embedding", "logits", "probability", "loss"],
                   help="Scoring method: embedding similarity, raw logits, "
                        "softmax probability, or cross-entropy loss")
    p.add_argument("--reduce_method", default="mean",
                   choices=["mean", "max", "min", "last"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from dnallm import DNAInference
    except ImportError:
        logger.error("DNALLM not found. Install with: pip install dnallm")
        sys.exit(1)

    inferencer = DNAInference(
        model_path=args.model_path,
        device=args.device,
        trust_remote_code=True,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info("Scoring %s (type=%s, reduce=%s)", args.input_file, args.score_type, args.reduce_method)
    scores = inferencer.scoring(
        inputs=args.input_file,
        reduce_method=args.reduce_method,
        score_type=args.score_type,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(scores, dict):
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=2, default=str)
    else:
        import pandas as pd
        df = pd.DataFrame(scores)
        df.to_csv(out_path, index=False)

    logger.info("Scores saved to %s", out_path)


if __name__ == "__main__":
    main()
