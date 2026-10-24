#!/usr/bin/env python3
"""Taxonomy prediction for fungal ITS sequences (HuggingFace remote-code model).

Loads FungiHelixSeek-Taxonomy via AutoModel + trust_remote_code, runs the six
classification heads (phylum -> species) and writes a CSV with per-level
predicted labels and softmax confidences.

Input: FASTA (optionally UNITE-style headers; only the sequence is used).
Output: CSV with one row per sequence: seq_id, seq_length, and
<{level}, {level}_confidence> for each of the six levels.
"""
from __future__ import annotations

import argparse
import sys
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

LEVELS = ["phylum", "class", "order", "family", "genus", "species"]


def parse_fasta(path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                parts = line[1:].split("|")
                seq_id = parts[0].strip() or (parts[2].strip() if len(parts) > 2 else "")
                header = seq_id.split()[0] if seq_id else f"seq_{len(records)}"
                chunks = []
            else:
                chunks.append(line.upper())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq_id, seq = self.records[idx]
        enc = self.tokenizer(seq, truncation=True, max_length=self.max_length)
        return {
            "seq_id": seq_id,
            "seq_len": len(seq),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }


def collate(features, pad_token_id):
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {
        "input_ids": torch.full((len(features), max_len), pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros((len(features), max_len), dtype=torch.long),
    }
    for i, f in enumerate(features):
        ids = torch.tensor(f["input_ids"], dtype=torch.long)
        mask = torch.tensor(f["attention_mask"], dtype=torch.long)
        batch["input_ids"][i, : len(ids)] = ids
        batch["attention_mask"][i, : len(mask)] = mask
    return batch


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="FungiHelixSeek-Taxonomy prediction")
    p.add_argument("--model_path", default="models/FungiHelixSeek-Taxonomy",
                   help="HF-format model directory (config.json + model.safetensors + remote code)")
    p.add_argument("--input_fasta", required=True, help="Input FASTA of ITS sequences")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--device", default="auto", help="cuda, cuda:0, mps, cpu, or auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=1024,
                   help="Token truncation length (must match training, 1024)")
    p.add_argument("--label_encoders", default=None,
                   help="label_encoders.json path (default: <model_path>/label_encoders.json)")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    encoders_path = Path(args.label_encoders) if args.label_encoders else Path(args.model_path) / "label_encoders.json"
    if not encoders_path.exists():
        raise FileNotFoundError(
            f"label_encoders.json not found at {encoders_path}; pass --label_encoders explicitly"
        )
    with open(encoders_path) as f:
        label_encoders = json.load(f)
    id2label = {lvl: {i: name for name, i in enc_map.items()} for lvl, enc_map in label_encoders.items()}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "huggingface"))
    from common import copy_remote_code_files

    copy_remote_code_files(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.to(device).eval()

    records = parse_fasta(args.input_fasta)
    if not records:
        raise ValueError(f"No sequences parsed from {args.input_fasta}")
    print(f"Loaded {len(records)} sequences from {args.input_fasta}")

    dataset = SequenceDataset(records, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda feats: collate(feats, tokenizer.pad_token_id),
    )

    rows: list[dict] = []
    offset = 0
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = [torch.softmax(logits, dim=-1).cpu() for logits in outputs["logits_list"]]
        preds = [prob.argmax(dim=-1).numpy() for prob in probs]
        confs = [prob.max(dim=-1).values.numpy() for prob in probs]

        for i in range(input_ids.shape[0]):
            rec = dataset[offset + i]
            row = {"seq_id": rec["seq_id"], "seq_length": rec["seq_len"]}
            for lvl_idx, lvl in enumerate(LEVELS):
                row[lvl] = id2label[lvl][int(preds[lvl_idx][i])]
                row[f"{lvl}_confidence"] = float(confs[lvl_idx][i])
            rows.append(row)
        offset += input_ids.shape[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {output}")


if __name__ == "__main__":
    main()
