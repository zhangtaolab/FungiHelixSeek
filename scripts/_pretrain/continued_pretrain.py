#!/usr/bin/env python3
"""Continued MLM pretraining using DNALLM's DNATrainer.

Requires: DNALLM >= 0.5.2 (pip install dnallm)
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek continued pretraining via DNALLM")
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--data_path", "-i", required=True, help="Raw text file, one sequence per line")
    p.add_argument("--output_dir", "-o", default="tmp/dnallm_pretrain")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer
        from dnallm.configuration import TaskConfig, TrainingConfig
    except ImportError:
        logger.error("DNALLM not found. Install with: pip install dnallm")
        sys.exit(1)

    logger.info("Loading model and tokenizer from %s", args.model_path)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, task_type="mask", trust_remote_code=True,
    )

    logger.info("Loading dataset from %s", args.data_path)
    dataset = DNADataset()
    dataset.load_local_data(args.data_path, format="text")

    task_config = TaskConfig(
        task_type="mask",
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        save_strategy="epoch",
    )

    dataset.encode_sequences(task="mask", tokenizer=tokenizer, max_length=args.max_length)

    trainer = DNATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset.dataset,
        task_config=task_config,
        training_config=training_config,
    )

    logger.info("Starting continued pretraining…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
