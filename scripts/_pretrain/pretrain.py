#!/usr/bin/env python3
"""Continued MLM pre-training for FungiHelixSeek.

Usage (single GPU)::

    python scripts/huggingface/pretrain.py \
        --model_path models/FungiHelixSeek \
        --tokenizer_path models/FungiHelixSeek \
        --data_path data/test_data/pretrain_example_data.txt \
        --output_dir tmp/pretrain_output

Multi-GPU with ``torchrun``::

    torchrun --nproc_per_node=4 scripts/huggingface/pretrain.py [same args]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from common import copy_remote_code_files, get_ddp_backend

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FungiHelixSeek continued MLM pre-training")
    p.add_argument("--model_path", "-m", required=True, help="Pretrained model directory or checkpoint")
    p.add_argument("--tokenizer_path", "-t", required=True, help="Tokenizer directory")
    p.add_argument("--data_path", "-i", required=True, help="Raw text file (one DNA sequence per line)")
    p.add_argument("--output_dir", "-o", default="tmp/pretrain_output", help="Output checkpoint directory")
    p.add_argument("--logging_dir", "-l", default=None, help="TensorBoard log dir (default: output_dir/logs)")
    p.add_argument("--from_pretrained", default="true", choices=["true", "false"],
                   help="true = continue from --model_path weights; false = random init (still uses config)")
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--bf16", default="true", choices=["true", "false"])
    p.add_argument("--gradient_checkpointing", default="false", choices=["true", "false"])
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_ratio", type=float, default=0.1, help="Fraction of data held out for eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--device_target", default="auto", choices=["auto", "cuda", "npu", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    copy_remote_code_files(args.model_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.logging_dir or str(out_dir / "logs")

    logger.info("Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    logger.info("Loading config from %s", args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.vocab_size = len(tokenizer)
    config.model_mode = "mlm"

    if args.from_pretrained == "true":
        logger.info("Loading pretrained weights from %s", args.model_path)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )
    else:
        logger.info("Random-init new model from config")
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    logger.info("Loading dataset from %s", args.data_path)
    raw = load_dataset("text", data_files=str(args.data_path), split="train")
    if args.max_train_samples:
        raw = raw.select(range(min(args.max_train_samples, len(raw))))

    def tokenize_fn(examples):
        seqs = [s.upper() for s in examples["text"]]
        return tokenizer(seqs, truncation=True, max_length=args.max_sequence_length)

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

    split = tokenized.train_test_split(
        test_size=args.eval_ratio, seed=args.seed,
        shuffle=True,
    )
    train_ds, eval_ds = split["train"], split["test"]
    logger.info("Train: %d sequences | Eval: %d sequences", len(train_ds), len(eval_ds))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8 if args.bf16 == "true" else None,
    )

    use_bf16 = args.bf16 == "true" and args.device_target in ("auto", "cuda", "npu")
    use_gc = args.gradient_checkpointing == "true"

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        logging_dir=log_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=use_bf16,
        gradient_checkpointing=use_gc,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training…")
    train_result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    for src in Path(args.model_path).glob("*.py"):
        dst = out_dir / src.name
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("Training complete. Checkpoints saved to %s", out_dir)


if __name__ == "__main__":
    main()
