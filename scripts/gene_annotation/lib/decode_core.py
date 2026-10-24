#!/usr/bin/env python3
"""Decode cached per-base probabilities into gene models on both strands."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ANNO_DIR = Path(__file__).resolve().parents[1]
if str(_ANNO_DIR) not in sys.path:
    sys.path.insert(0, str(_ANNO_DIR))

from decoder import gff_writer as GW  # noqa: E402
from decoder import viterbi as V  # noqa: E402
from . import predict_core as PC  # noqa: E402

N_LABELS = PC.N_LABELS
N_PHASE = PC.N_PHASE


def load_fasta(path) -> dict[str, str]:
    return PC.read_fasta(Path(path))


def _array(z, key, length, width=None, fill=0.0):
    if key in z.files:
        arr = np.asarray(z[key], dtype=np.float32)
        arr = arr[:length]
    else:
        arr = np.full((length,) if width is None else (length, width),
                      fill, dtype=np.float32)
    return arr


def _decode_strand(seq, p17, phase, bnd, len_prior, rescue):
    path, _ = V.viterbi_decode(seq, p17, phase, bnd, len_prior=len_prior)
    genes = [g for g in V.decode_to_genes(path) if g.get("cds")]
    if rescue and len(genes) > 1:
        genes = V.rescue_split_genes(seq, genes, p17, phase, bnd,
                                     len_prior=len_prior)
    return [g for g in genes if g.get("cds") and g.get("end") is not None]


def _mirror(gene, length):
    def flip(interval):
        start, end = interval
        return (length - 1 - end, length - 1 - start)

    mirrored = {
        "start": length - 1 - gene["end"],
        "end": length - 1 - gene["start"],
        "cds": sorted(flip(iv) for iv in gene["cds"]),
        "introns": sorted(flip(iv) for iv in gene["introns"]),
        "u5": flip(gene["u3"]) if gene.get("u3") else None,
        "u3": flip(gene["u5"]) if gene.get("u5") else None,
        "strand": "-",
    }
    return mirrored


def _gene_score(gene, p17):
    start = max(0, gene["start"])
    end = min(len(p17) - 1, gene["end"])
    if end < start:
        return 0.0
    genic = p17[start:end + 1, 1:].sum(axis=1)
    return float(genic.mean())


def _select_non_overlapping(genes, p17):
    ranked = sorted(genes, key=lambda g: _gene_score(g, p17), reverse=True)
    kept: list[dict] = []
    for gene in ranked:
        if all(gene["end"] < k["start"] or gene["start"] > k["end"]
               for k in kept):
            kept.append(gene)
    return sorted(kept, key=lambda g: g["start"])


def _decode_one(task, len_prior=True, rescue=True):
    name, seq, _kind, path, _extra = task
    length = len(seq)
    with np.load(path) as z:
        fwd17 = _array(z, "fwd_probs17", length, N_LABELS)
        fwd_phase = _array(z, "fwd_phase", length, N_PHASE, 1.0 / N_PHASE)
        fwd_bnd = _array(z, "fwd_bnd", length, None, 0.0)
        rev17 = _array(z, "rev_probs17", length, N_LABELS)
        rev_phase = _array(z, "rev_phase", length, N_PHASE, 1.0 / N_PHASE)
        rev_bnd = _array(z, "rev_bnd", length, None, 0.0)

    forward = _decode_strand(seq, fwd17, fwd_phase, fwd_bnd, len_prior, rescue)
    for gene in forward:
        gene["strand"] = "+"

    rc = PC.reverse_complement(seq)
    reverse = _decode_strand(rc, rev17, rev_phase, rev_bnd, len_prior, rescue)
    reverse = [_mirror(g, length) for g in reverse]

    return _select_non_overlapping(forward + reverse, fwd17)


def gene_to_gff_lines(seqid, gene, gid, source="FungiHelixSeek", strand="+"):
    return GW.gene_to_gff_lines(seqid, gene, gid, source=source, strand=strand)


__all__ = ["load_fasta", "gene_to_gff_lines", "_decode_one"]
