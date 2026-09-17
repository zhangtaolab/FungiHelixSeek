#!/usr/bin/env python3
"""End-to-end smoke tests for FungiHelixSeek HuggingFace scripts.

Run from the repository root with the environment's Python interpreter::

    cd <FungiHelixSeek repo root>
    PYTHONPATH=scripts/huggingface python tests/test_smoke.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "scripts" / "huggingface"),
       "HF_ENDPOINT": "https://hf-mirror.com"}

PASSED = 0
FAILED = 0
SKIPPED = 0


def run_cmd(desc: str, cmd: list[str], timeout: int = 300):
    global PASSED, FAILED
    print(f"\n{'='*60}")
    print(f"TEST: {desc}")
    print(f"{'='*60}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=ENV,
                           cwd=str(REPO_ROOT))
        if r.returncode == 0:
            print(f"  ✓ PASSED")
            PASSED += 1
        else:
            print(f"  ✗ FAILED (exit {r.returncode})")
            print(r.stderr[-500:] if r.stderr else "(no stderr)")
            FAILED += 1
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT ({timeout}s)")
        FAILED += 1
        return False


def skip_cmd(desc: str):
    global SKIPPED
    print(f"\n{'='*60}")
    print(f"TEST: {desc}")
    print(f"{'='*60}")
    print(f"  → SKIPPED (required files not present)")
    SKIPPED += 1


def main():
    model = str(REPO_ROOT / "models" / "FungiHelixSeek")
    tmp = REPO_ROOT / "tmp" / "test_smoke"
    tmp.mkdir(parents=True, exist_ok=True)

    run_cmd("1. Verify install", [
        PY, "scripts/huggingface/verify_install.py",
        "--model_path", model, "--device", "auto",
    ], timeout=60)

    run_cmd("2. Inference (mask recovery)", [
        PY, "scripts/huggingface/inference.py",
        "--model_path", model,
        "--input_file", str(REPO_ROOT / "data" / "test_data" / "inference_example.jsonl"),
        "--output_file", str(tmp / "mask_recovery.jsonl"),
        "--top_k", "3", "--device", "auto", "--torch_dtype", "float32",
    ], timeout=120)

    pretrain_script = REPO_ROOT / "scripts" / "_pretrain" / "pretrain.py"
    pretrain_data = REPO_ROOT / "scripts" / "_pretrain" / "pretrain_example_data.txt"
    if pretrain_script.is_file() and pretrain_data.is_file():
        run_cmd("3. Pretrain (50 samples, 1 epoch)", [
            PY, str(pretrain_script),
            "--model_path", model,
            "--tokenizer_path", str(model),
            "--data_path", str(pretrain_data),
            "--output_dir", str(tmp / "pretrain"),
            "--max_sequence_length", "128",
            "--num_train_epochs", "1",
            "--per_device_train_batch_size", "4",
            "--bf16", "true",
            "--max_train_samples", "50",
            "--save_steps", "1000", "--logging_steps", "10",
        ], timeout=300)
    else:
        skip_cmd("3. Pretrain (50 samples, 1 epoch)")

    run_cmd("4. Finetune (40 samples, 1 epoch)", [
        PY, "scripts/huggingface/finetune.py",
        "--pretrained_model", model,
        "--dataset_dir", str(REPO_ROOT / "data" / "test_data" / "finetune_example_data"),
        "--output_dir", str(tmp / "finetune"),
        "--tokens_length", "128",
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "4",
        "--bf16", "true",
        "--max_train_samples", "40",
        "--max_eval_samples", "10",
        "--max_test_samples", "10",
    ], timeout=300)

    run_cmd("5. Predict (using fine-tuned model)", [
        PY, "scripts/huggingface/predict.py",
        "--model_path", str(tmp / "finetune"),
        "--input_file", str(REPO_ROOT / "data" / "test_data" / "predict_example.csv"),
        "--output_file", str(tmp / "predictions.csv"),
        "--batch_size", "8", "--device", "auto",
    ], timeout=60)

    print(f"\n{'='*60}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"{'='*60}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
