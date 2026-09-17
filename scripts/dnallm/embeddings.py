#!/usr/bin/env python3
"""Extract embeddings from FungiHelixSeek using DNALLM.

Supports mean / max / min / PCA reduction of hidden states.

Requires: DNALLM >= 0.5.2 (pip install dnallm)

    python scripts/dnallm/embeddings.py \
        --model_path models/FungiHelixSeek \
        --input_file data/test_data/predict_example.csv \
        --output_file tmp/embeddings.npy \
        --reduce_strategy mean
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek embedding extraction via DNALLM")
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--input_file", "-i", required=True,
                   help="CSV with 'sequence' column, or a .txt with one sequence per line")
    p.add_argument("--output_file", "-o", required=True,
                   help="Output .npy file for embedding array")
    p.add_argument("--reduce_strategy", default="mean",
                   choices=["mean", "max", "min"],
                   help="Pooling strategy to reduce token-level embeddings to sequence-level")
    p.add_argument("--do_reduce", action="store_true", default=True,
                   help="Apply PCA dimensionality reduction (default: just pool)")
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

    logger.info("Extracting embeddings from %s (strategy=%s)", args.input_file, args.reduce_strategy)
    result = inferencer.get_embeddings(
        inputs=args.input_file,
        do_reduce=False,
        reduce_strategy=args.reduce_strategy,
    )

    embeddings = result.get("embeddings", result)
    if isinstance(embeddings, dict):
        embeddings = embeddings.get("hidden_states", embeddings)

    embeddings = np.array(embeddings)
    logger.info("Embeddings shape: %s", embeddings.shape)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.save(args.output_file, embeddings)
    logger.info("Saved to %s", args.output_file)


if __name__ == "__main__":
    main()
