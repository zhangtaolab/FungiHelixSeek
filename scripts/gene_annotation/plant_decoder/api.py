"""In-memory Plant decoder API.

``decode_contig`` decodes one contig from already-computed 17-class probability
arrays (the existing Fungi cache format) into :class:`~plant_decoder.model.Gene`
objects.  It is intentionally self-contained: no Plant checkout, Plant CLI, or
sibling module import is required at runtime.

Coordinate conventions
----------------------
Returned genes use half-open genomic intervals ``[start, end)`` as defined by
:class:`~plant_decoder.model.Exon`; the GFF writer performs the 1-based
inclusive conversion.  Forward probabilities (``p17``) are expected in forward
genomic order.  Reverse probabilities (``p17_r``) come from the Neural/CRF
cache in reverse-complement orientation; ``rev_arrays_rc=True`` (the default)
reverses them exactly once so the Plant forward-genomic decoder sees the same
orientation it does when reading its own NPZ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Gene
from .orf import correct_all_genes, merge_genes_by_intron_prob
from .postprocess import (
    detect_all_utrs,
    filter_low_confidence_single_exons,
    filter_to_min_cds_length,
    filter_to_min_gene_length,
    filter_to_valid_genes,
)
from .priors import resolve_priors
from .repair import rc, repair_genes
from .viterbi import decode_viterbi
from .viterbi_utils import N_LABELS, aggregate_probs17_to_features

SUPPORTED_DECODERS = ("repair", "viterbi", "orf")

__all__ = ["DecoderConfig", "SUPPORTED_DECODERS", "decode_contig"]


@dataclass
class DecoderConfig:
    """Configuration for :func:`decode_contig`.

    Defaults are the validated Plant ``viterbi+repair`` benchmark settings.
    ``phase``/``bnd`` arrays are accepted by :func:`decode_contig` only for
    legacy call-signature compatibility; Plant decoding ignores them.
    """

    decoder: str = "repair"
    prior: str = "fungi"
    transition_npz: str | None = None
    intergenic_boost: float = 0.0
    viterbi_alpha: float = 0.0
    genic_threshold: float = 0.25
    min_cds_run: int = 5
    single_exon_prob: float = 0.7
    single_exon_min_cds: int = 200
    repair_w5: int = 300
    repair_w3: int = 300
    repair_start_rule: str = "nearest"
    orf_max_shift: int = 3
    rev_arrays_rc: bool = True


def _validate_probabilities(
    arr: object,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    """Validate one ``(L, 17)`` probability array and return float32.

    Arrays must be two-dimensional, have exactly ``N_LABELS`` columns, match the
    contig length, and contain only finite numeric values.  The input is never
    modified.
    """
    probs = np.asarray(arr)
    if probs.ndim != 2 or probs.shape[1] != N_LABELS:
        raise ValueError(
            f"{name} must have shape (L, {N_LABELS}); got {probs.shape}"
        )
    if probs.shape[0] != length:
        raise ValueError(
            f"{name} length {probs.shape[0]} does not match sequence length "
            f"{length}"
        )
    if not np.issubdtype(probs.dtype, np.number):
        raise ValueError(f"{name} must contain numeric probabilities")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{name} contains non-finite values")
    probs32 = probs.astype(np.float32, copy=False)
    if not np.all(np.isfinite(probs32)):
        raise ValueError(f"{name} contains values outside float32 range")
    return probs32


def _decode_strand(
    contig: str,
    feature_probs: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    strand: str,
    gene_count: int,
    config: DecoderConfig,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray,
) -> tuple[list[Gene], int]:
    """Decode one strand with the shared Plant Viterbi routine."""
    return decode_viterbi(
        contig,
        feature_probs,
        labels,
        probabilities,
        strand,
        gene_count,
        transition_matrix=transition_matrix,
        initial_probs=initial_probs,
        viterbi_alpha=config.viterbi_alpha,
        min_cds_run=config.min_cds_run,
        genic_threshold=config.genic_threshold,
        intergenic_boost=config.intergenic_boost,
    )


def decode_contig(
    contig,
    seq,
    p17,
    phase,
    bnd,
    p17_r=None,
    p_phase_r=None,
    p_bnd_r=None,
    config=None,
    rev_arrays_rc=None,
) -> list[Gene]:
    """Decode one contig from cached 17-class probability arrays.

    Parameters
    ----------
    contig
        Contig name written onto every returned :class:`Gene`.
    seq
        Forward-genomic contig sequence.  Its length must match ``p17``.
    p17
        Forward ``(L, 17)`` probabilities in forward genomic order.
    phase, bnd
        Legacy HMM arrays accepted for call-signature compatibility.  Plant
        decoding does not use phase or boundary emissions; these are ignored.
    p17_r
        Optional reverse-strand ``(L, 17)`` probabilities.  By default they are
        reverse-complement oriented and are reversed once into forward genomic
        order.
    p_phase_r, p_bnd_r
        Legacy reverse HMM arrays; ignored for the same reason as ``phase`` and
        ``bnd``.
    config
        :class:`DecoderConfig` instance.  ``None`` uses the benchmark defaults.
    rev_arrays_rc
        Optional override for ``config.rev_arrays_rc``.  ``True`` reverses
        ``p17_r`` exactly once; ``False`` treats it as already forward-genomic.

    Returns
    -------
    list[Gene]
        Genes in Plant's forward-then-reverse decoding order after the standard
        minimum-CDS, minimum-gene, valid-gene and single-exon filters, with 5'
        and 3' UTR spans detected last.
    """
    del phase, bnd, p_phase_r, p_bnd_r  # HMM-only; Plant methods ignore them

    if config is None:
        config = DecoderConfig()
    elif not isinstance(config, DecoderConfig):
        raise TypeError(
            "config must be a DecoderConfig instance or None; "
            f"got {type(config).__name__}"
        )

    decoder = str(config.decoder).strip().lower()
    if decoder not in SUPPORTED_DECODERS:
        raise ValueError(
            f"Unsupported decoder {config.decoder!r}; expected one of "
            f"{SUPPORTED_DECODERS}"
        )

    if not isinstance(seq, str):
        raise TypeError(f"seq must be a string; got {type(seq).__name__}")
    length = len(seq)

    if rev_arrays_rc is None:
        reverse_once = bool(config.rev_arrays_rc)
    else:
        reverse_once = bool(rev_arrays_rc)

    forward_probs = _validate_probabilities(p17, length=length, name="p17")
    reverse_probs = None
    if p17_r is not None:
        reverse_probs = _validate_probabilities(
            p17_r, length=length, name="p17_r"
        )
        if reverse_once:
            # Cached reverse probabilities are reverse-complement oriented;
            # the Plant decoder works in forward genomic coordinates.
            reverse_probs = np.ascontiguousarray(reverse_probs[::-1])

    transition_matrix, initial_probs = resolve_priors(
        config.prior, config.transition_npz
    )

    if length == 0:
        return []

    forward_labels = np.argmax(forward_probs, axis=1)
    forward_features = aggregate_probs17_to_features(forward_probs)

    all_genes, gene_count = _decode_strand(
        contig,
        forward_features,
        forward_labels,
        forward_probs,
        "+",
        0,
        config,
        transition_matrix,
        initial_probs,
    )

    reverse_labels = None
    if reverse_probs is not None:
        reverse_labels = np.argmax(reverse_probs, axis=1)
        reverse_features = aggregate_probs17_to_features(reverse_probs)
        reverse_genes, gene_count = _decode_strand(
            contig,
            reverse_features,
            reverse_labels,
            reverse_probs,
            "-",
            gene_count,
            config,
            transition_matrix,
            initial_probs,
        )
        all_genes = all_genes + reverse_genes

    # Plant's operation order: method-specific correction first, then filters,
    # then UTR detection.  ``reverse_probs`` is already forward-genomic here.
    probs_for_reverse = forward_probs if reverse_probs is None else reverse_probs
    labels_for_reverse = (
        forward_labels if reverse_labels is None else reverse_labels
    )

    if decoder == "orf":
        all_genes = correct_all_genes(
            all_genes,
            seq,
            forward_probs,
            probs_for_reverse,
            orf_max_shift=config.orf_max_shift,
        )
        all_genes = merge_genes_by_intron_prob(
            all_genes, forward_probs, probs_for_reverse
        )
    elif decoder == "repair":
        repair_genes(
            all_genes,
            seq,
            rc_seq=rc(seq),
            w5=config.repair_w5,
            w3=config.repair_w3,
            start_rule=config.repair_start_rule,
        )

    all_genes = filter_to_min_cds_length(all_genes)
    all_genes = filter_to_min_gene_length(all_genes)
    all_genes = filter_to_valid_genes(all_genes)
    all_genes = filter_low_confidence_single_exons(
        all_genes,
        forward_probs,
        probs_for_reverse,
        min_prob=config.single_exon_prob,
        min_cds_len=config.single_exon_min_cds,
    )

    detect_all_utrs(all_genes, forward_labels, labels_for_reverse)

    return all_genes
