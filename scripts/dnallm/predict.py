#!/usr/bin/env python3
"""Run predictions with a fine-tuned FungiHelixSeek model using DNALLM.

Requires: DNALLM >= 0.5.2 (pip install dnallm)
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek prediction via DNALLM")
    p.add_argument("--model_path", "-m", required=True, help="Fine-tuned model directory")
    p.add_argument("--input_file", "-i", required=True, help="CSV with 'sequence' column")
    p.add_argument("--output_file", "-o", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="auto")
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
    )

    logger.info("Running predictions on %s", args.input_file)
    results = inferencer.infer_file(
        args.input_file,
        output_file=args.output_file,
        sequence_column="sequence",
    )

    logger.info("Predictions saved to %s", args.output_file)
    if isinstance(results, dict) and "metrics" in results:
        logger.info("Metrics: %s", results["metrics"])


if __name__ == "__main__":
    main()
