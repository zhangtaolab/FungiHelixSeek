#!/usr/bin/env python3
"""In-silico mutagenesis scan with FungiHelixSeek using DNALLM.

Performs saturation mutagenesis: mutates each position to every possible
nucleotide and measures the effect on model output. Useful for identifying
functionally important positions (e.g., promoters, enhancers, splice sites).

Requires: DNALLM >= 0.5.2 (pip install dnallm)

    python scripts/dnallm/mutagenesis.py \
        --model_path tmp/finetune_output \
        --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
        --output_dir tmp/mutagenesis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek saturation mutagenesis via DNALLM")
    p.add_argument("--model_path", "-m", required=True,
                   help="Fine-tuned classification model (recommended)")
    p.add_argument("--sequence", "-s", required=True,
                   help="DNA sequence to scan")
    p.add_argument("--output_dir", "-o", default="tmp/mutagenesis")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from dnallm import Mutagenesis
    except ImportError:
        logger.error("DNALLM not found. Install with: pip install dnallm")
        sys.exit(1)

    mut = Mutagenesis(
        model_path=args.model_path,
        device=args.device,
        trust_remote_code=True,
        batch_size=args.batch_size,
    )

    logger.info("Running saturation mutagenesis on sequence (len=%d)", len(args.sequence))
    results = mut.evaluate(sequence=args.sequence)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(mut, "plot"):
        try:
            mut.plot(save_path=str(out_dir / "mutagenesis_plot.png"))
            logger.info("Plot saved to %s/mutagenesis_plot.png", out_dir)
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    import json
    with open(out_dir / "mutagenesis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s/mutagenesis_results.json", out_dir)


if __name__ == "__main__":
    main()
