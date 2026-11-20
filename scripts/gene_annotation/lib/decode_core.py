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


def _hmm_array(arr, length, width, fill):
    """Coerce one cached-style HMM array to float32 without changing values.

    ``None`` becomes the same neutral default the npz loader uses (uniform
    phase, zero boundary), and arrays longer than ``length`` are truncated so
    they line up with the sequence exactly as the ``_decode_one`` loader does.
    """
    if arr is None:
        shape = (length,) if width is None else (length, width)
        return np.full(shape, fill, dtype=np.float32)
    return np.asarray(arr, dtype=np.float32)[:length]


def decode_hmm_arrays(name, seq, p17, phase, bnd, p17_r=None, p_phase_r=None,
                      p_bnd_r=None, len_prior=True, rescue=True):
    """Decode one contig with the legacy HMM path, returning dict gene models.

    This is the exact ``_decode_strand`` + ``_mirror`` +
    ``_select_non_overlapping`` pipeline used by :func:`_decode_one`, but taking
    arrays directly instead of an npz path.  Reverse-strand arrays are the
    cache's reverse-complement-oriented arrays: the reverse-complement sequence
    is decoded, then coordinates are mirrored back into forward genomic order.
    ``name`` is accepted for symmetry with :func:`decode_contig`; legacy dict
    genes do not carry a contig field.
    """
    del name  # legacy dict gene models do not store the contig name
    length = len(seq)
    fwd17 = _hmm_array(p17, length, N_LABELS, 0.0)
    fwd_phase = _hmm_array(phase, length, N_PHASE, 1.0 / N_PHASE)
    fwd_bnd = _hmm_array(bnd, length, None, 0.0)

    forward = _decode_strand(seq, fwd17, fwd_phase, fwd_bnd, len_prior, rescue)
    for gene in forward:
        gene["strand"] = "+"

    if p17_r is None:
        return _select_non_overlapping(forward, fwd17)

    rev17 = _hmm_array(p17_r, length, N_LABELS, 0.0)
    rev_phase = _hmm_array(p_phase_r, length, N_PHASE, 1.0 / N_PHASE)
    rev_bnd = _hmm_array(p_bnd_r, length, None, 0.0)

    rc = PC.reverse_complement(seq)
    reverse = _decode_strand(rc, rev17, rev_phase, rev_bnd, len_prior, rescue)
    reverse = [_mirror(g, length) for g in reverse]

    return _select_non_overlapping(forward + reverse, fwd17)


def decode_contig(name, seq, p17, phase, bnd, p17_r=None, p_phase_r=None,
                  p_bnd_r=None, decoder="repair", len_prior=True, rescue=True,
                  **config_kwargs):
    """Decode one contig with either the Plant or legacy HMM decoder.

    The shared entry point used by both the GUI and the CLI.

    Parameters
    ----------
    name, seq, p17, phase, bnd, p17_r, p_phase_r, p_bnd_r
        Contig name/sequence plus forward and reverse ``(L, 17)`` probability
        arrays and legacy HMM phase/boundary arrays.  Reverse arrays are in the
        cache's reverse-complement orientation.
    decoder
        ``"hmm"`` uses :func:`decode_hmm_arrays` (dict gene models).  ``"repair"``,
        ``"viterbi"`` and ``"orf"`` use the vendored Plant decoder (``Gene``
        objects).  Any other value raises ``ValueError``.
    len_prior, rescue
        Legacy HMM toggles.  Accepted, but ignored by the Plant decoders.
    **config_kwargs
        Extra :class:`plant_decoder.DecoderConfig` fields (``prior``,
        ``intergenic_boost``, ``repair_w5``, ...).  Ignored for ``"hmm"``.

    Returns
    -------
    list[Gene] | list[dict]
        ``Gene`` objects for the Plant decoders, legacy dicts for ``"hmm"``.
    """
    decoder_name = str(decoder).strip().lower()
    if decoder_name == "hmm":
        return decode_hmm_arrays(
            name, seq, p17, phase, bnd,
            p17_r=p17_r, p_phase_r=p_phase_r, p_bnd_r=p_bnd_r,
            len_prior=len_prior, rescue=rescue,
        )

    # Lazy import keeps the Plant decoder's numba dependency out of HMM-only use.
    from plant_decoder.api import DecoderConfig, decode_contig as _plant_decode

    config = DecoderConfig(decoder=decoder_name, **config_kwargs)
    return _plant_decode(
        name, seq, p17, phase, bnd,
        p17_r=p17_r, p_phase_r=p_phase_r, p_bnd_r=p_bnd_r,
        config=config,
    )


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

    return decode_hmm_arrays(
        name, seq, fwd17, fwd_phase, fwd_bnd,
        p17_r=rev17, p_phase_r=rev_phase, p_bnd_r=rev_bnd,
        len_prior=len_prior, rescue=rescue,
    )


def gene_to_gff_lines(seqid, gene, gid, source="FungiHelixSeek", strand="+"):
    return GW.gene_to_gff_lines(seqid, gene, gid, source=source, strand=strand)


__all__ = [
    "decode_contig",
    "decode_hmm_arrays",
    "gene_to_gff_lines",
    "load_fasta",
    "_decode_one",
]
