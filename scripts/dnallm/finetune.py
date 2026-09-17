#!/usr/bin/env python3
"""Fine-tune FungiHelixSeek for sequence classification using DNALLM.

Requires: DNALLM >= 0.5.2 (pip install dnallm)
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="FungiHelixSeek fine-tuning via DNALLM")
    p.add_argument("--model_path", "-m", required=True)
    p.add_argument("--dataset_dir", "-d", required=True,
                   help="Directory with train*.csv, dev*.csv, test*.csv")
    p.add_argument("--output_dir", "-o", default="tmp/dnallm_finetune")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from dnallm import DNADataset, DNATrainer
        from dnallm.models.model import load_model_and_tokenizer
    except ImportError:
        logger.error("DNALLM not found. Install with: pip install dnallm")
        sys.exit(1)

    from pathlib import Path

    data_dir = Path(args.dataset_dir)
    train_file = next(data_dir.glob("train*.csv"))
    dev_file = next(data_dir.glob("dev*.csv"))
    test_file = next(data_dir.glob("test*.csv"))

    logger.info("Train: %s | Dev: %s | Test: %s", train_file, dev_file, test_file)

    train_ds = DNADataset()
    train_ds.load_local_data(str(train_file), format="csv", text_column="sequence",
                             label_column="label")
    val_ds = DNADataset()
    val_ds.load_local_data(str(dev_file), format="csv", text_column="sequence",
                           label_column="label")

    unique_labels = sorted(set(train_ds.dataset["label"]))
    n_labels = len(unique_labels)
    task_type = "binary" if n_labels == 2 else "multiclass"
    logger.info("Task: %s | num_labels: %d", task_type, n_labels)

    from dnallm.configuration import TaskConfig, TrainingConfig

    task_config = TaskConfig(
        task_type=task_type,
        num_labels=n_labels,
        max_length=args.max_length,
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        metric_for_best_model="f1" if task_type == "binary" else "f1_macro",
        save_strategy="epoch",
    )

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        task_type=task_type,
        num_labels=n_labels,
        trust_remote_code=True,
    )

    train_ds.encode_sequences(task="SequenceClassification", tokenizer=tokenizer,
                              max_length=args.max_length)
    val_ds.encode_sequences(task="SequenceClassification", tokenizer=tokenizer,
                            max_length=args.max_length)

    trainer = DNATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds.dataset,
        eval_dataset=val_ds.dataset,
        task_config=task_config,
        training_config=training_config,
    )

    logger.info("Starting fine-tuning…")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Evaluating on test set…")
    test_ds = DNADataset()
    test_ds.load_local_data(str(test_file), format="csv", text_column="sequence",
                            label_column="label")
    test_ds.encode_sequences(task="SequenceClassification", tokenizer=tokenizer,
                             max_length=args.max_length)
    results = trainer.evaluate(test_ds.dataset)
    logger.info("Test results: %s", results)
    logger.info("Done. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
