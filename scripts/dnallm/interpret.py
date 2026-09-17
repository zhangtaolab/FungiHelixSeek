#!/usr/bin/env python3
"""Model interpretability analysis with FungiHelixSeek using DNALLM.

Uses Captum attribution methods (Integrated Gradients, DeepLift, Occlusion,
GradientShap, etc.) to identify which nucleotide positions drive the model's
predictions. Generates attribution plots.

Requires: DNALLM >= 0.5.2 (pip install dnallm)

    python scripts/dnallm/interpret.py \
        --model_path tmp/finetune_output \
        --sequence ATCGGCTAAGCCTGTCAGTTACCGATCG \
        --method integrated_gradients \
        --output_dir tmp/interpret
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek interpretability via DNALLM (Captum)")
    p.add_argument("--model_path", "-m", required=True,
                   help="Fine-tuned model (classification recommended)")
    p.add_argument("--sequence", "-s", required=True,
                   help="DNA sequence to interpret")
    p.add_argument("--method", default="integrated_gradients",
                   choices=["integrated_gradients", "deep_lift", "deep_lift_shap",
                            "gradient_shap", "occlusion", "saliency",
                            "input_x_gradient", "lime", "feature_ablation",
                            "kernelshap"],
                   help="Captum attribution method")
    p.add_argument("--output_dir", "-o", default="tmp/interpret")
    p.add_argument("--device", default="auto")
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from dnallm import DNAInterpret
    except ImportError:
        logger.error("DNALLM not found. Install with: pip install dnallm")
        sys.exit(1)

    interpreter = DNAInterpret(
        model_path=args.model_path,
        device=args.device,
        trust_remote_code=True,
        max_length=args.max_length,
    )

    logger.info("Running %s on sequence (len=%d)", args.method, len(args.sequence))
    results = interpreter.interpret(
        sequence=args.sequence,
        method=args.method,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(interpreter, "plot"):
        try:
            interpreter.plot(save_path=str(out_dir / f"{args.method}_plot.png"))
            logger.info("Plot saved to %s/%s_plot.png", out_dir, args.method)
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    import json
    with open(out_dir / f"{args.method}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s/%s_results.json", out_dir, args.method)


if __name__ == "__main__":
    main()
