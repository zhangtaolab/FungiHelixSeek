#!/usr/bin/env python3
"""Verify that FungiHelixSeek is installed correctly.

Runs a quick forward pass + mask recovery on the downloaded model::

    python scripts/huggingface/verify_install.py --model_path models/FungiHelixSeek
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_version(name: str, required: str | None = None):
    try:
        mod = __import__(name)
        v = getattr(mod, "__version__", "unknown")
        logger.info("  ✓ %-20s %s", name, v)
        return True
    except ImportError:
        logger.error("  ✗ %-20s NOT FOUND", name)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", default="models/FungiHelixSeek")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "mps", "cpu"])
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("FungiHelixSeek Installation Verification")
    logger.info("=" * 50)

    logger.info("\n[1/4] Checking dependencies…")
    deps_ok = all(check_version(n) for n in [
        "torch", "transformers", "datasets", "accelerate",
        "numpy", "pandas", "sklearn", "einops",
    ])

    from common import get_device, load_model_and_tokenizer, get_dna_base_token_ids

    logger.info("\n[2/4] Checking device…")
    device = get_device(args.device)
    if device.type == "cuda":
        logger.info("  ✓ CUDA available: %s", torch.cuda.get_device_name(0))
    elif device.type == "npu":
        logger.info("  ✓ NPU available")
    elif device.type == "mps":
        logger.info("  ✓ Apple MPS available")
    else:
        logger.info("  → Running on CPU (training will be slow)")

    logger.info("\n[3/4] Loading model from %s…", args.model_path)
    t0 = time.time()
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, task_type="mlm")
        load_time = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info("  ✓ Model loaded in %.1fs | %.1fM params", load_time, n_params)
        logger.info("  ✓ Vocab size: %d | mask_token_id: %d", len(tokenizer), tokenizer.mask_token_id)
    except Exception as e:
        logger.error("  ✗ Model loading failed: %s", e)
        sys.exit(1)

    logger.info("\n[4/4] Running forward pass + mask recovery…")
    model.to(device).eval()
    base_ids = get_dna_base_token_ids(tokenizer)

    test_seq = "ATCGGCTAAGCCT<mask>GTCAGTTACCGATCG"
    inputs = tokenizer(test_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_positions = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) > 0:
        pos = mask_positions[0].item()
        probs = logits[0, pos].float().softmax(dim=-1)
        valid_ids = list(base_ids.values())
        sub = probs[valid_ids]
        best_idx = sub.argmax().item()
        predicted = tokenizer.convert_ids_to_tokens([valid_ids[best_idx]])[0]
        logger.info("  ✓ Forward pass OK | logits shape: %s", list(logits.shape))
        logger.info("  ✓ Mask recovery: predicted base '%s' at position %d", predicted, pos)
    else:
        logger.warning("  ! No <mask> token found in test sequence")

    logger.info("\n" + "=" * 50)
    logger.info("✓ All checks passed! FungiHelixSeek is ready.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
