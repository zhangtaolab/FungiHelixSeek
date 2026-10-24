#!/usr/bin/env python3
"""Performance benchmark for FungiHelixSeek.

Sweeps task × batch_size × gradient_checkpointing and reports throughput
(samples/s, tokens/s) + peak VRAM.  Also estimates wall-clock time for a
1 GiB pretraining corpus.

::

    python scripts/huggingface/benchmark.py \
        --model_path models/FungiHelixSeek \
        --output_file tmp/benchmark_results.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from common import (
    copy_remote_code_files,
    device_synchronize,
    empty_device_cache,
    get_device,
    get_device_name,
    load_model_and_tokenizer,
    parse_torch_dtype,
    peak_memory_gb,
    reset_peak_memory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DNA_ALPHABET = "ATCG"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FungiHelixSeek performance benchmark")
    p.add_argument("--model_path", "-m", default="models/FungiHelixSeek")
    p.add_argument("--output_file", "-o", default="tmp/benchmark_results.csv")
    p.add_argument("--seq_length", type=int, default=512,
                   help="Fixed sequence length for benchmarking")
    p.add_argument("--seq_lengths", default=None,
                   help="Comma-separated sequence lengths (overrides --seq_length). E.g. 128,512,1024,2048,4096")
    p.add_argument("--batch_sizes", default="1,2,4,8,16",
                   help="Comma-separated batch sizes to test")
    p.add_argument("--tasks", default="pretrain,inference",
                   help="Comma-separated tasks: pretrain,finetune,inference")
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--measure_steps", type=int, default=10)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "npu", "mps", "cpu"])
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"],
                   help="Model dtype for benchmarking (default: bfloat16, matching real pretraining)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def generate_sequences(n: int, length: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(DNA_ALPHABET, k=length)) for _ in range(n)]


@torch.no_grad()
def benchmark_inference(model, tokenizer, sequences, batch_size, device, seq_len, warmup, iters):
    model.eval()
    all_batches = [sequences[i:i+batch_size] for i in range(0, len(sequences), batch_size)]

    for _ in range(warmup):
        batch = all_batches[0]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=seq_len, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        _ = model(**enc)

    device_synchronize(device)

    reset_peak_memory(device)
    t0 = time.perf_counter()
    total_tokens = 0
    n_done = 0
    for _ in range(iters):
        for batch in all_batches:
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=seq_len, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
            total_tokens += enc["input_ids"].numel()
            n_done += len(batch)

    device_synchronize(device)
    dt = time.perf_counter() - t0
    vram = peak_memory_gb(device)

    return {
        "samples_per_sec": n_done / dt,
        "tokens_per_sec": total_tokens / dt,
        "peak_vram_gb": vram,
        "wall_time_sec": dt,
    }


def benchmark_pretrain(model, tokenizer, sequences, batch_size, device, seq_len,
                       warmup, iters, gradient_checkpointing=False):
    from transformers import AutoModelForMaskedLM

    model.train()
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable({"use_reentrant": False})

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )

    enc = tokenizer(sequences, truncation=True, max_length=seq_len, padding="max_length",
                    return_tensors="pt")
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])

    all_batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for _ in range(warmup):
        input_ids, attn = all_batches[0]
        input_ids, attn = input_ids.to(device), attn.to(device)
        labels = input_ids.clone()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    reset_peak_memory(device)
    device_synchronize(device)

    t0 = time.perf_counter()
    total_tokens = 0
    n_done = 0
    for _ in range(iters):
        for batch in all_batches:
            input_ids, attn = batch
            input_ids, attn = input_ids.to(device), attn.to(device)
            labels = input_ids.clone()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_tokens += input_ids.numel()
            n_done += input_ids.size(0)

    device_synchronize(device)
    dt = time.perf_counter() - t0
    vram = peak_memory_gb(device)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    return {
        "samples_per_sec": n_done / dt,
        "tokens_per_sec": total_tokens / dt,
        "peak_vram_gb": vram,
        "wall_time_sec": dt,
    }


def estimate_pretrain_time(tokens_per_sec: float, dataset_size_gb: float, avg_seq_len: int):
    avg_bytes_per_char = 1
    total_chars = dataset_size_gb * 1024**3 / avg_bytes_per_char
    total_tokens = total_chars
    seconds = total_tokens / tokens_per_sec
    hours = seconds / 3600
    return {"dataset_tokens": int(total_tokens), "estimated_hours": round(hours, 1)}


def main():
    args = parse_args()
    device = get_device(args.device)
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.seq_lengths:
        seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]
    else:
        seq_lengths = [args.seq_length]

    logger.info("Device: %s (%s)", device, get_device_name(device))
    logger.info("Seq lengths: %s | Batch sizes: %s | Tasks: %s", seq_lengths, batch_sizes, tasks)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, task_type="mlm",
        torch_dtype=parse_torch_dtype(args.torch_dtype),
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model params: %.1fM", n_params)

    results: List[Dict] = []

    for seq_len in seq_lengths:
        n_seqs = max(batch_sizes) * (args.warmup_steps + args.measure_steps + 5)
        sequences = generate_sequences(n_seqs, seq_len, args.seed)
        logger.info("=== Seq length: %d | Generated %d sequences ===", seq_len, n_seqs)

        for task in tasks:
            for bs in batch_sizes:
                for gc_on in ([False, True] if task in ("pretrain", "finetune") else [False]):
                    gc_tag = "on" if gc_on else "off"
                    label = f"{task} | bs={bs} | gc={gc_tag} | seq={seq_len}"
                    logger.info("Running: %s", label)

                    try:
                        if task == "inference":
                            r = benchmark_inference(
                                model, tokenizer, sequences, bs, device, seq_len,
                                args.warmup_steps, args.measure_steps,
                            )
                        elif task in ("pretrain", "finetune"):
                            r = benchmark_pretrain(
                                model, tokenizer, sequences, bs, device, seq_len,
                                args.warmup_steps, args.measure_steps, gc_on,
                            )
                        else:
                            continue
                    except torch.cuda.OutOfMemoryError:
                        logger.warning("OOM for %s — skipping", label)
                        gc.collect()
                        empty_device_cache(device)
                        results.append({
                            "task": task, "batch_size": bs, "seq_len": seq_len,
                            "gradient_checkpointing": gc_on, "status": "OOM",
                            "samples_per_sec": "", "tokens_per_sec": "", "peak_vram_gb": "",
                        })
                        continue
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("OOM for %s — skipping", label)
                            gc.collect()
                            empty_device_cache(device)
                            results.append({
                                "task": task, "batch_size": bs, "seq_len": seq_len,
                                "gradient_checkpointing": gc_on, "status": "OOM",
                                "samples_per_sec": "", "tokens_per_sec": "", "peak_vram_gb": "",
                            })
                            continue
                        raise

                    logger.info(
                        "  → %.1f samples/s | %.0f tokens/s | %.2f GiB VRAM",
                        r["samples_per_sec"], r["tokens_per_sec"], r["peak_vram_gb"],
                    )
                    results.append({
                        "task": task, "batch_size": bs, "seq_len": seq_len,
                        "gradient_checkpointing": gc_on, "status": "OK",
                        "samples_per_sec": round(r["samples_per_sec"], 2),
                        "tokens_per_sec": round(r["tokens_per_sec"], 1),
                        "peak_vram_gb": round(r["peak_vram_gb"], 3),
                    })

                    gc.collect()
                    empty_device_cache(device)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "batch_size", "seq_len", "gradient_checkpointing",
            "status", "samples_per_sec", "tokens_per_sec", "peak_vram_gb",
        ])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", out_path)

    ok = [r for r in results if r["status"] == "OK" and r["task"] == "pretrain" and not r["gradient_checkpointing"]]
    if ok:
        logger.info("=" * 60)
        logger.info("1 GiB pretraining estimates (by seq_len × batch_size):")
        for sl in sorted(set(r["seq_len"] for r in ok)):
            for bs in sorted(set(r["batch_size"] for r in ok)):
                match = [r for r in ok if r["seq_len"] == sl and r["batch_size"] == bs]
                if not match:
                    continue
                tps = float(match[0]["tokens_per_sec"])
                est = estimate_pretrain_time(tps, 1.0, sl)
                logger.info("  seq=%d bs=%d → %.0f tok/s → ~%.1f hours for 1GiB",
                            sl, bs, tps, est["estimated_hours"])
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
