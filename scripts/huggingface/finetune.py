#!/usr/bin/env python3
"""Fine-tune FungiHelixSeek for sequence classification.

Auto-detects binary / multi-class / regression from the CSV label column.
Expects a dataset directory containing ``train*.csv``, ``dev*.csv``,
``test*.csv`` (columns: ``sequence``, ``label``; ``sp`` optional)::

    python scripts/huggingface/finetune.py \
        --pretrained_model models/FungiHelixSeek \
        --dataset_dir data/test_data/finetune_example_data \
        --output_dir tmp/finetune_output \
        --tokens_length 512
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import (
    build_compute_metrics,
    copy_remote_code_files,
    get_ddp_backend,
    infer_task_type,
    load_tokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FungiHelixSeek fine-tuning")
    p.add_argument("--pretrained_model", "-m", required=True)
    p.add_argument("--dataset_dir", "-d", required=True)
    p.add_argument("--output_dir", "-o", default="tmp/finetune_output")
    p.add_argument("--tokens_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--classification_pooling", default="max",
                   choices=["last", "cls", "mean", "max"])
    p.add_argument("--bf16", default="true", choices=["true", "false"])
    p.add_argument("--gradient_checkpointing", default="false", choices=["true", "false"])
    p.add_argument("--metric_for_best_model", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_target", default="auto", choices=["auto", "cuda", "npu", "mps", "cpu"])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    return p.parse_args()


def find_csv(directory: Path, prefix: str) -> Path:
    matches = sorted(directory.glob(f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {prefix}*.csv in {directory}")
    if len(matches) > 1:
        logger.warning("Multiple %s*.csv found, using %s", prefix, matches[0])
    return matches[0]


def main():
    args = parse_args()
    from transformers import AutoModelForSequenceClassification

    data_dir = Path(args.dataset_dir)
    train_file = find_csv(data_dir, "train")
    dev_file = find_csv(data_dir, "dev")
    test_file = find_csv(data_dir, "test")
    logger.info("Train: %s | Dev: %s | Test: %s", train_file, dev_file, test_file)

    copy_remote_code_files(args.pretrained_model)
    tokenizer = load_tokenizer(args.pretrained_model)

    ds = load_dataset("csv", data_files={
        "train": str(train_file), "validation": str(dev_file), "test": str(test_file),
    })

    train_labels = [float(r["label"]) for r in ds["train"]]
    task_type, num_labels = infer_task_type(train_labels)
    logger.info("Task type: %s | num_labels: %d", task_type, num_labels)

    label_keys = None
    if task_type != "regression":
        unique_labels = sorted(set(int(l) for l in train_labels))
        label_keys = {i: str(l) for i, l in enumerate(unique_labels)}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "classification_pooling"):
        model.config.classification_pooling = args.classification_pooling

    if task_type != "regression" and label_keys:
        model.config.id2label = label_keys
        model.config.label2id = {v: k for k, v in label_keys.items()}

    score = getattr(model, "score", None)
    if score is not None and hasattr(score, "weight"):
        torch.nn.init.xavier_uniform_(score.weight)

    def tokenize_fn(examples):
        seqs = [s.upper() for s in examples["sequence"]]
        return tokenizer(seqs, truncation=True, max_length=args.tokens_length)

    cols_to_remove = [c for c in ds["train"].column_names if c not in ("label",)]
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(
            range(min(args.max_train_samples, len(tokenized["train"]))))
    if args.max_eval_samples:
        tokenized["validation"] = tokenized["validation"].select(
            range(min(args.max_eval_samples, len(tokenized["validation"]))))

    compute_metrics = build_compute_metrics(task_type)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = math.ceil(
        len(tokenized["train"])
        / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
        * args.num_train_epochs
    )
    eval_steps = max(total_steps // 10, 1)
    save_steps = eval_steps

    metric_name = args.metric_for_best_model
    if metric_name is None:
        metric_name = "r2" if task_type == "regression" else "f1" if task_type == "binary" else "f1_macro"
    greater_better = metric_name not in ("eval_loss", "mse")

    use_bf16 = args.bf16 == "true" and args.device_target in ("auto", "cuda")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing == "true",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater_better,
        logging_steps=max(eval_steps // 2, 1),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
    )
    if int(os.environ.get("LOCAL_RANK", -1)) != -1:
        training_args.ddp_backend = get_ddp_backend()
        training_args.ddp_find_unused_parameters = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    for src in Path(args.pretrained_model).glob("*.py"):
        dst = out_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    test_ds = tokenized["test"]
    if args.max_test_samples:
        test_ds = test_ds.select(range(min(args.max_test_samples, len(test_ds))))

    test_result = trainer.predict(test_ds)
    test_metrics = {k: float(v) for k, v in test_result.metrics.items() if k.startswith("test_")}
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
    logger.info("Fine-tuning complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
