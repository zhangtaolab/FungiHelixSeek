#!/usr/bin/env python3
"""Sliding-window inference for the FungiHelixSeek-Anno token classifier.

The checkpoint ships only the 17-class BILOU classifier; the 4-class reading
frame head is a frozen linear probe (``phase_head_fungi.npz``) applied to the
final hidden states. Per-base outputs are averaged over overlapping windows.
"""

from __future__ import annotations

import gzip
import importlib
import shutil
from pathlib import Path

import numpy as np
import torch

WINDOW = 8192
STRIDE = 2048

BOUNDARY_LABELS = (1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16)
N_LABELS = 17
N_PHASE = 4

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def read_fasta(path: Path) -> dict[str, str]:
    """Read a plain or gzip-compressed FASTA into ``{name: sequence}``."""
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            elif name is not None:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks).upper()
    return seqs


def write_fasta(seqs: dict[str, str], path: Path, width: int = 60) -> None:
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _copy_remote_code_files(model_path: Path) -> Path:
    """Mirror ``scripts/huggingface/common.copy_remote_code_files`` so the
    dynamic-module loader can resolve ``mlp.py`` before it is cached."""
    from transformers.utils import HF_MODULES_CACHE

    cache_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / model_path.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "__init__.py").touch()
    pycache = cache_dir / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)
    for py_file in model_path.glob("*.py"):
        shutil.copy2(py_file, cache_dir / py_file.name)
    importlib.invalidate_caches()
    return cache_dir


def load_model(model_path: Path, device: torch.device, phase_head: Path | None):
    """Load the token classifier plus an optional external phase head.

    Returns ``(model, tokenizer, has_phase)``.
    """
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_path = Path(model_path)
    _copy_remote_code_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, do_lower_case=False)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path), trust_remote_code=True)
    model.to(device).eval()

    weight = bias = None
    builtin = getattr(model, "phase_head", None)
    if builtin is not None and hasattr(builtin, "weight"):
        weight, bias = builtin.weight.detach().float(), builtin.bias.detach().float()
    elif phase_head is not None and Path(phase_head).is_file():
        z = np.load(str(phase_head))
        weight = torch.from_numpy(z["weight"]).float()
        bias = torch.from_numpy(z["bias"]).float()

    has_phase = weight is not None and bias is not None
    if has_phase:
        model._anno_phase_head = (
            weight.to(device), bias.to(device))
    return model, tokenizer, has_phase


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _phase_probs(model, hidden: torch.Tensor) -> torch.Tensor:
    weight, bias = model._anno_phase_head
    return torch.softmax(hidden @ weight.t() + bias, dim=-1)


@torch.inference_mode()
def predict_contig(model, tokenizer, seq: str, device: torch.device,
                   batch_size: int, has_phase: bool, window: int = WINDOW,
                   stride: int = STRIDE, overlap_avg: bool = True):
    """Per-base 17-class probabilities, 4-class phase probabilities and
    boundary scores for one contig."""
    length = len(seq)
    starts = _window_starts(length, window, stride)
    acc17 = np.zeros((length, N_LABELS), dtype=np.float64)
    acc_phase = np.zeros((length, N_PHASE), dtype=np.float64)
    acc_bnd = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    use_amp = device.type == "cuda"

    for batch_start in range(0, len(starts), batch_size):
        batch = starts[batch_start:batch_start + batch_size]
        windows = [seq[s:s + window] for s in batch]
        enc = tokenizer(windows, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=has_phase)

        probs = out.logits.float().softmax(dim=-1).cpu().numpy()
        bnd = probs[:, :, list(BOUNDARY_LABELS)].sum(axis=-1)
        phase = None
        if has_phase:
            hidden = out.hidden_states[-1].float()
            phase = _phase_probs(model, hidden).float().cpu().numpy()

        for row, start in enumerate(batch):
            n = len(windows[row])
            if overlap_avg:
                acc17[start:start + n] += probs[row, :n]
                acc_bnd[start:start + n] += bnd[row, :n]
                counts[start:start + n] += 1.0
                if phase is not None:
                    acc_phase[start:start + n] += phase[row, :n]
            else:
                acc17[start:start + n] = probs[row, :n]
                acc_bnd[start:start + n] = bnd[row, :n]
                counts[start:start + n] = 1.0
                if phase is not None:
                    acc_phase[start:start + n] = phase[row, :n]

    counts = np.maximum(counts, 1.0)
    acc17 /= counts[:, None]
    acc_bnd /= counts
    acc_phase /= counts[:, None]

    if not has_phase:
        acc_phase[:] = 1.0 / N_PHASE
    return acc17.astype(np.float32), acc_phase.astype(np.float32), \
        acc_bnd.astype(np.float32)


__all__ = [
    "read_fasta", "write_fasta", "reverse_complement", "load_model",
    "predict_contig", "BOUNDARY_LABELS", "N_LABELS", "N_PHASE",
]
