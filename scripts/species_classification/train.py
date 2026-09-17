#!/usr/bin/env python3
"""Fine-tune FungiHelixSeek for 6-level taxonomic classification of fungal ITS barcodes.

Trains the FungiTaxonomyForTaxonomicClassification architecture (HelixSeek
backbone + 6 linear heads) with Hierarchical Label Smoothing (HLS, MycoAI
Eq. 1-4) and inverse-frequency class weights, mirroring the released
FungiHelixSeek-Taxonomy training recipe.

Input FASTA headers follow the UNITE convention:
    >accession|k__Fungi;p__...;c__...;o__...;f__...;g__...;s__...|SHxxxx.xxFU
Unknown ranks ("s__?") are masked with ignore_index=-1.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoConfig, AutoTokenizer

LEVELS = ["phylum", "class", "order", "family", "genus", "species"]
LEVEL_ABBR = ["p__", "c__", "o__", "f__", "g__", "s__"]
UNKNOWN_INT = -1


def parse_fasta(path):
    records = []
    header, chunks = None, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        else:
            chunks.append(line.upper())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def parse_taxonomy(header):
    tax_str = header.split("|")[1] if "|" in header else header
    tax = {}
    for field in tax_str.split(";"):
        for abbr, name in zip(LEVEL_ABBR, LEVELS):
            if field.startswith(abbr):
                value = field[3:]
                tax[name] = UNKNOWN_INT if value in ("", "?") else value
    return [tax.get(lv, UNKNOWN_INT) for lv in LEVELS]


def build_label_encoders(records):
    encoders = {lv: {} for lv in LEVELS}
    for header, _ in records:
        for i, label in enumerate(parse_taxonomy(header)):
            if label == UNKNOWN_INT:
                continue
            encoders[LEVELS[i]].setdefault(label, len(encoders[LEVELS[i]]))
    return encoders


def encode(records, encoders):
    labels = np.full((len(records), 6), UNKNOWN_INT, dtype=np.int64)
    for i, (header, _) in enumerate(records):
        for j, label in enumerate(parse_taxonomy(header)):
            if label == UNKNOWN_INT:
                continue
            labels[i, j] = encoders[LEVELS[j]].get(label, UNKNOWN_INT)
    return labels


def build_hierarchy_matrices(labels, num_classes):
    """Child->parent matrices: each child row is one-hot on its most frequent
    parent (uniform when never observed), matching the released recipe."""
    matrices = []
    for lvl in range(len(LEVELS) - 1):
        n_child, n_parent = num_classes[lvl + 1], num_classes[lvl]
        rows = torch.as_tensor(labels, dtype=torch.long)
        child, parent = rows[:, lvl + 1], rows[:, lvl]
        valid = (child != UNKNOWN_INT) & (parent != UNKNOWN_INT)
        flat = torch.bincount(
            child[valid] * n_parent + parent[valid], minlength=n_child * n_parent
        ).float().reshape(n_child, n_parent)
        m = torch.full((n_child, n_parent), 1.0 / n_parent)
        seen = flat.sum(dim=1) > 0
        m[seen] = torch.zeros_like(m[seen])
        m[seen, flat[seen].argmax(dim=1)] = 1.0
        matrices.append(m)
    return matrices


def compute_class_weights(labels, num_classes, strength=0.5):
    weights = []
    for lvl in range(len(num_classes)):
        counts = torch.bincount(
            torch.as_tensor(labels[labels[:, lvl] >= 0, lvl]), minlength=num_classes[lvl]
        ).float()
        w = 1.0 / counts.clamp(min=1) ** strength
        weights.append(w / w.mean())
    return weights


class HierarchicalLabelSmoothingLoss(nn.Module):
    def __init__(self, hierarchy_matrices, num_classes, smoothing=None, class_weights=None):
        super().__init__()
        self.num_levels = len(num_classes)
        self.num_classes = num_classes
        self.ignore_index = UNKNOWN_INT
        self.smoothing = list(smoothing) if smoothing else [0.02] * 5 + [0.0]
        self.class_weights = class_weights

        col_norm = [(M / M.sum(dim=0, keepdim=True).clamp(min=1)).t() for M in hierarchy_matrices]
        self.transitions = {}
        for i in range(self.num_levels - 1):
            t = col_norm[i]
            self.transitions[(i, i + 1)] = t
            for l in range(i + 2, self.num_levels):
                t = t @ col_norm[l - 1]
                self.transitions[(i, l)] = t

    def forward(self, logits_list, target_labels):
        bs, device = target_labels.shape[0], target_labels.device
        eps_u = self.smoothing[5]
        total = torch.zeros((), device=device)
        n_valid = 0

        for lvl in range(self.num_levels):
            known = target_labels[:, lvl] != self.ignore_index
            if not known.any():
                continue
            n_cls = self.num_classes[lvl]
            target = torch.full((bs, n_cls), eps_u / max(n_cls - 1, 1), device=device)
            eps_hier = sum(self.smoothing[:lvl])
            target[known, target_labels[known, lvl]] = 1.0 - eps_u - eps_hier
            for i in range(lvl):
                t = self.transitions.get((i, lvl))
                if t is None:
                    continue
                t = t.to(device)
                known_i = target_labels[:, i] != self.ignore_index
                if known_i.any():
                    target[known_i] += self.smoothing[i] * t[target_labels[known_i, i]]
            log_probs = F.log_softmax(logits_list[lvl][known], dim=-1)
            loss = -(target[known] * log_probs).sum(dim=-1)
            if self.class_weights is not None:
                w = self.class_weights[lvl].to(device)
                loss = loss * w[target_labels[known, lvl]]
            total = total + loss.mean()
            n_valid += 1
        return total / max(n_valid, 1)


class ITSClassificationDataset(Dataset):
    def __init__(self, records, labels, tokenizer, max_length):
        self.records = records
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.records[idx][1], truncation=True, max_length=self.max_length
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": self.labels[idx]}


def collate(batch, pad_token_id=1):
    labels = torch.tensor(np.array([b["labels"] for b in batch]), dtype=torch.long)
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attention_mask[i, :n] = torch.tensor(b["attention_mask"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_bf16=False):
    model.eval()
    total, count = 0.0, 0
    correct = torch.zeros(6, dtype=torch.long)
    seen = torch.zeros(6, dtype=torch.long)
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]
        with amp_ctx:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            batch_loss = loss_fn(out["logits_list"], labels)
        total += batch_loss.item() * len(labels)
        count += len(labels)
        for lvl in range(6):
            valid = labels[:, lvl] != UNKNOWN_INT
            if valid.any():
                preds = out["logits_list"][lvl].argmax(-1)
                correct[lvl] += (preds[valid] == labels[valid, lvl]).sum().cpu()
                seen[lvl] += valid.sum().cpu()
    model.train()
    acc = {LEVELS[i]: (correct[i] / seen[i]).item() if seen[i] else None for i in range(6)}
    return total / max(count, 1), acc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train_fasta", required=True)
    p.add_argument("--valid_fasta", default=None)
    p.add_argument("--model_template", default="models/FungiHelixSeek-Taxonomy",
                   help="dir providing the FungiTaxonomyForTaxonomicClassification architecture code")
    p.add_argument("--init_backbone_from", default=None,
                   help="base FungiHelixSeek dir whose backbone weights seed training "
                        "(skipped when the template already carries trained weights)")
    p.add_argument("--label_encoders", default=None,
                   help="existing label_encoders.json to keep class ids stable "
                        "(default: rebuild from the training FASTA)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="validation batch size (default: same as --batch_size)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="trade compute for memory (recommended for 32 GB GPUs at batch >= 16)")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the forward pass")
    p.add_argument("--device", default="auto")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_records = parse_fasta(args.train_fasta)
    if args.label_encoders:
        encoders = json.load(open(args.label_encoders))
    else:
        encoders = build_label_encoders(train_records)
    num_classes = [len(encoders[lv]) for lv in LEVELS]
    print(f"train={len(train_records)} classes={dict(zip(LEVELS, num_classes))}")

    train_labels = encode(train_records, encoders)
    valid_loader = None
    if args.valid_fasta:
        valid_records = parse_fasta(args.valid_fasta)
        valid_labels = encode(valid_records, encoders)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "huggingface"))
    from common import copy_remote_code_files

    copy_remote_code_files(args.model_template)
    tokenizer = AutoTokenizer.from_pretrained(args.model_template, trust_remote_code=True)
    train_ds = ITSClassificationDataset(train_records, train_labels, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    if args.valid_fasta:
        valid_ds = ITSClassificationDataset(valid_records, valid_labels, tokenizer, args.max_length)
        valid_loader = DataLoader(valid_ds, batch_size=args.eval_batch_size or args.batch_size,
                                  collate_fn=collate, num_workers=args.num_workers)

    config = AutoConfig.from_pretrained(args.model_template, trust_remote_code=True)
    config.num_classes = num_classes
    config.dropout = args.dropout
    model = AutoModel.from_config(config, trust_remote_code=True)
    if args.init_backbone_from:
        backbone = AutoModel.from_pretrained(args.init_backbone_from, trust_remote_code=True)
        model.backbone.load_state_dict(backbone.state_dict())
        del backbone
        print(f"backbone initialised from {args.init_backbone_from}")
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled")

    hierarchy = build_hierarchy_matrices(train_labels, num_classes)
    weights = compute_class_weights(train_labels, num_classes)
    loss_fn = HierarchicalLabelSmoothingLoss(hierarchy, num_classes, class_weights=weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total_steps = (len(train_loader)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0

    model.train()
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with amp_ctx:
                outputs = model(**batch)
                loss = loss_fn(outputs["logits_list"], labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            step += 1
        msg = f"epoch {epoch}: train_loss={epoch_loss / len(train_loader):.4f}"

        if valid_loader is not None:
            val_loss, acc = evaluate(model, valid_loader, loss_fn, device, use_bf16=args.bf16)
            msg += f" val_loss={val_loss:.4f} acc={acc}"
            species_acc = acc.get("species") or 0.0
            if species_acc > best_acc:
                best_acc = species_acc
                model.save_pretrained(out)
                json.dump(encoders, open(out / "label_encoders.json", "w"))
                msg += " (best -> saved)"
        print(msg)

    if valid_loader is None:
        model.save_pretrained(out)
        json.dump(encoders, open(out / "label_encoders.json", "w"))
    tokenizer.save_pretrained(out)
    for py in Path(args.model_template).glob("*.py"):
        shutil.copy2(py, out / py.name)
    print(f"model saved to {out} (best species acc: {best_acc:.4f})")


if __name__ == "__main__":
    main()
