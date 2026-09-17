#!/usr/bin/env python3
"""Shared utilities for FungiHelixSeek HuggingFace scripts.

Handles the critical mlp.py cache workaround, device detection
(CUDA / CPU), model loading, and metrics computation.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)

DNA_BASES: Tuple[str, ...] = ("A", "T", "C", "G", "N")

try:
    import torch_npu  # noqa: F401
    HAS_TORCH_NPU = True
except ImportError:
    HAS_TORCH_NPU = False


def copy_remote_code_files(model_path: str) -> Path:
    r"""Pre-copy every ``*.py`` from the model directory into the HF
    ``transformers_modules`` cache.

    HuggingFace's dynamic-module loader only copies files directly referenced
    by ``auto_map`` and files reachable through a *shallow* relative-import
    scan.      For the HelixSeek architecture the chain is::

        model.py  →  layers.py  →  mlp.py

    The loader discovers ``mlp`` as a dependency of ``layers.py`` but tries to
    *open* ``mlp.py`` inside the cache to scan *its* imports — before it has
    ever been copied there — causing ``FileNotFoundError``.  We sidestep the
    whole problem by copying every ``*.py`` upfront.
    """
    from transformers.utils import HF_MODULES_CACHE

    model_name = os.path.basename(os.path.normpath(model_path))
    safe_name = model_name.replace(".", "_")
    cache_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / safe_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "__init__.py").touch()

    pycache = cache_dir / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    copied: List[str] = []
    for py_file in Path(model_path).glob("*.py"):
        shutil.copy2(py_file, cache_dir / py_file.name)
        copied.append(py_file.name)

    if copied:
        importlib.invalidate_caches()
        logger.debug("Copied %d remote-code files: %s", len(copied), sorted(copied))
    return cache_dir


def get_device(device_target: str = "auto") -> torch.device:
    """Return a ``torch.device`` for ``auto`` / ``cuda`` / ``npu`` / ``cpu``."""
    if device_target == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if HAS_TORCH_NPU and torch.npu.is_available():
            return torch.device("npu")
        return torch.device("cpu")

    if device_target == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")

    if device_target == "npu":
        if not HAS_TORCH_NPU:
            raise RuntimeError(
                "NPU requested but torch_npu is not installed. "
                "Install with: pip install torch-npu"
            )
        return torch.device("npu")

    return torch.device(device_target)


def get_device_count() -> int:
    """Number of accelerator devices visible (GPU or NPU)."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if HAS_TORCH_NPU and torch.npu.is_available():
        return torch.npu.device_count()
    return 0


def get_ddp_backend() -> str:
    """``'nccl'`` on CUDA, ``'hccl'`` on NPU, ``'gloo'`` otherwise."""
    if torch.cuda.is_available():
        return "nccl"
    if HAS_TORCH_NPU and torch.npu.is_available():
        return "hccl"
    return "gloo"


def parse_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_str.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from {list(mapping)}")
    return mapping[key]


def load_tokenizer(model_path: str, **kwargs):
    from transformers import AutoTokenizer

    copy_remote_code_files(model_path)
    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **kwargs
    )


def load_model(
    model_path: str,
    task_type: str = "mlm",
    *,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """Load a FungiHelixSeek model head with the ``mlp.py`` fix applied.

    ``task_type`` selects the HF ``Auto*`` class:

    * ``"mlm"``                      → ``AutoModelForMaskedLM``
    * ``"sequence_classification"`` → ``AutoModelForSequenceClassification``
    * ``"token_classification"``    → ``AutoModelForTokenClassification``
    """
    from transformers import (
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
    )

    copy_remote_code_files(model_path)

    cls_map = {
        "mlm": AutoModelForMaskedLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "token_classification": AutoModelForTokenClassification,
    }
    cls = cls_map.get(task_type)
    if cls is None:
        raise ValueError(f"task_type must be one of {list(cls_map)}, got '{task_type}'")

    load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    load_kwargs.update(kwargs)
    return cls.from_pretrained(model_path, **load_kwargs)


def load_model_and_tokenizer(
    model_path: str,
    task_type: str = "mlm",
    *,
    torch_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """Convenience: ``(model, tokenizer)`` in one call."""
    tok = load_tokenizer(model_path)
    mdl = load_model(model_path, task_type, torch_dtype=torch_dtype, **kwargs)
    return mdl, tok


def get_dna_base_token_ids(tokenizer) -> Dict[str, int]:
    """Map each DNA base → single token ID (filters multi-token bases)."""
    result: Dict[str, int] = {}
    for base in DNA_BASES:
        ids = tokenizer.encode(base, add_special_tokens=False)
        if len(ids) == 1:
            result[base] = ids[0]
    return result


_MASK_RE = re.compile(r"<mask>|<MASK>|\[mask\]|\[MASK\]")


def normalize_mask_tokens(sequence: str, mask_token: str) -> str:
    """Replace any common mask spelling (``<mask>`` / ``[MASK]`` / …) with
    ``mask_token``, uppercasing all other characters."""
    placeholder = "\x00MASK\x00"
    seq = _MASK_RE.sub(placeholder, sequence)
    seq = seq.upper()
    seq = seq.replace(placeholder, mask_token)
    return seq


def infer_task_type(labels: List[Any]) -> Tuple[str, int]:
    """Return ``(task_type, num_labels)`` from a list of label values.

    * float with non-integer values → ``"regression"``
    * exactly 2 unique integers    → ``"binary"``
    * > 2 unique integers           → ``"multi"``
    """
    unique = sorted(set(labels))
    is_regression = any(isinstance(l, float) and not float(l).is_integer() for l in labels)
    if is_regression:
        return "regression", 1

    n = len(unique)
    if n <= 1:
        return "regression", 1
    if n == 2:
        return "binary", 2
    return "multi", n


def build_compute_metrics(task_type: str):
    """Return a ``compute_metrics(eval_pred)`` callable using HuggingFace ``evaluate``."""
    import evaluate
    from scipy.stats import spearmanr as _spearmanr
    from sklearn.metrics import r2_score as _r2_score

    _accuracy = evaluate.load("accuracy")
    _f1 = evaluate.load("f1")
    _precision = evaluate.load("precision")
    _recall = evaluate.load("recall")
    _mcc = evaluate.load("matthews_correlation")
    try:
        _auroc = evaluate.load("roc_auc")
    except Exception:
        _auroc = None

    def _fn(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits)
        labels = np.asarray(labels).reshape(-1)

        if task_type == "regression":
            if logits.shape[-1] == 1:
                preds = logits.squeeze(-1).reshape(-1)
            else:
                preds = logits.reshape(-1)
            return {
                "mse": float(np.mean((preds - labels) ** 2)),
                "r2": float(_r2_score(labels, preds)),
                "spearmanr": float(_spearmanr(preds, labels)[0]),
            }

        preds = np.argmax(logits, axis=-1).reshape(-1)
        result = {}
        result.update(_accuracy.compute(predictions=preds, references=labels))
        result.update(_mcc.compute(predictions=preds, references=labels))

        if task_type == "binary":
            result.update(_f1.compute(predictions=preds, references=labels))
            result.update(_precision.compute(predictions=preds, references=labels))
            result.update(_recall.compute(predictions=preds, references=labels))
            if _auroc is not None:
                try:
                    probs = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
                    result.update(_auroc.compute(prediction_scores=probs, references=labels))
                except Exception:
                    pass
        else:
            for avg in ("macro", "micro"):
                result[f"f1_{avg}"] = float(
                    _f1.compute(predictions=preds, references=labels, average=avg)["f1"]
                )
                result[f"precision_{avg}"] = float(
                    _precision.compute(predictions=preds, references=labels, average=avg)["precision"]
                )
                result[f"recall_{avg}"] = float(
                    _recall.compute(predictions=preds, references=labels, average=avg)["recall"]
                )

        return result

    return _fn


def peak_memory_gb(device: torch.device) -> float:
    """Peak memory allocated on ``device`` in GiB (0 for CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    if device.type == "npu" and HAS_TORCH_NPU:
        return torch.npu.max_memory_allocated(device) / (1024**3)
    return 0.0


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == "npu" and HAS_TORCH_NPU:
        torch.npu.reset_peak_memory_stats(device)
        torch.npu.empty_cache()


def device_synchronize(device: torch.device) -> None:
    """Block until all queued ops on ``device`` finish.

    Essential for accurate wall-clock timing on asynchronous accelerators
    (e.g. CUDA). No-op for CPU.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu" and HAS_TORCH_NPU:
        torch.npu.synchronize()


def empty_device_cache(device: torch.device) -> None:
    """Release the framework caching allocator's free memory on ``device``."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu" and HAS_TORCH_NPU:
        torch.npu.empty_cache()


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "npu" and HAS_TORCH_NPU:
        return torch.npu.get_device_name(device)
    return "CPU"
