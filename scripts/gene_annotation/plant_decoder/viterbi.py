"""5-class Viterbi genic filtering and CDS interval formation.

Ported from PlantHelixSeek's ``pred_to_gff3.decode_viterbi`` and
``_merge_adjacent_cds_viterbi``.  Unlike the source module this port never
reads a module-global transition matrix: callers pass ``transition_matrix``
and ``initial_probs`` explicitly, as produced by :mod:`plant_decoder.priors`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .model import Exon, Gene
from .viterbi_utils import (
    INTERVAL_ENTITY_CLASSES,
    convert_bilou_to_entity_labels,
    convert_entity_labels_to_intervals,
    viterbi_decode,
)

logger = logging.getLogger(__name__)

# Benchmark defaults for the valid viterbi+repair configuration.
VITERBI_ALPHA = 0.0
MIN_CDS_RUN = 5
GENIC_THRESHOLD = 0.25
INTERGENIC_BOOST = 0.0

MAX_MERGE_GAP = 50000
MERGE_GAP_INTRON_PROB = 0.30

CDS_ENTITY = 5
INTRON_ENTITY = 3


def decode_viterbi(
    chrom: str,
    feature_probs: np.ndarray,
    bilou_labels: np.ndarray,
    bilou_probs: np.ndarray,
    strand: str,
    gc: int,
    *,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray | None = None,
    viterbi_alpha: float = VITERBI_ALPHA,
    min_cds_run: int = MIN_CDS_RUN,
    genic_threshold: float = GENIC_THRESHOLD,
    intergenic_boost: float = INTERGENIC_BOOST,
) -> tuple[list[Gene], int]:
    """Use CRF BILUO labels for exon boundaries with 5-class Viterbi filtering.

    Parameters
    ----------
    chrom, strand, gc
        Contig name, strand, and the running gene counter.  Gene IDs continue
        from ``gc + 1`` and the updated counter is returned.
    feature_probs
        ``(L, 5)`` feature probabilities in ``[intergenic, intron,
        five_prime_utr, cds, three_prime_utr]`` order.
    bilou_labels
        ``(L,)`` integer 17-class BILUO labels supplying CDS boundaries.
    bilou_probs
        ``(L, 17)`` 17-class probabilities.  Accepted for interface parity with
        Plant; the decoder derives boundaries from ``bilou_labels`` and genic
        filtering from ``feature_probs``.
    transition_matrix, initial_probs
        Explicit 5x5 transition matrix and optional initial distribution.  A
        ``None`` initial uses the matrix's stationary distribution (Plant
        convention).  No module-global priors are consulted.
    viterbi_alpha, min_cds_run, genic_threshold, intergenic_boost
        Benchmark defaults: ``0.0``, ``5``, ``0.25``, ``0.0``.

    Returns
    -------
    tuple[list[Gene], int]
        Decoded genes (half-open CDS exons) and the updated gene counter.
    """
    del bilou_probs  # interface parity; boundaries come from bilou_labels

    probs = np.asarray(feature_probs, dtype=np.float64).copy()

    if intergenic_boost > 0:
        probs[:, 0] = np.minimum(probs[:, 0] + intergenic_boost * 0.05, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
    elif intergenic_boost < 0:
        probs[:, 0] = np.maximum(probs[:, 0] + intergenic_boost * 0.05, 1e-10)
        probs = probs / probs.sum(axis=1, keepdims=True)

    viterbi_labels = viterbi_decode(
        emission_probs=probs,
        transition_matrix=transition_matrix,
        initial_probs=initial_probs,
        alpha=viterbi_alpha,
    )

    genic_mask = viterbi_labels != 0
    genic_fraction = float(genic_mask.mean())
    use_genic_filter = genic_fraction > 0.01

    if not use_genic_filter:
        genic_mask = np.ones(len(probs), dtype=bool)

    entity_labels = convert_bilou_to_entity_labels(bilou_labels)

    bilou_intervals = convert_entity_labels_to_intervals(
        labels=entity_labels, class_groups=INTERVAL_ENTITY_CLASSES
    )

    cds_intervals = bilou_intervals[bilou_intervals["entity"] == CDS_ENTITY]

    genes: list[Gene] = []
    for _, row in cds_intervals.iterrows():
        s, e = int(row["start"]), int(row["stop"]) + 1
        if e - s < min_cds_run:
            continue
        genic_frac = float(genic_mask[s:e].mean())
        if genic_frac < genic_threshold:
            continue
        gc += 1
        genes.append(Gene(f"gene_{gc:06d}", chrom, strand, [Exon(s, e, strand)]))

    genes = _merge_adjacent_cds_viterbi(genes, bilou_intervals, probs, min_cds_run)
    return genes, gc


def _merge_adjacent_cds_viterbi(
    genes: list[Gene],
    intervals: pd.DataFrame,
    probs: np.ndarray,
    min_cds_run: int,
) -> list[Gene]:
    """Merge adjacent CDS intervals connected by intron labels."""
    if len(genes) <= 1:
        return genes

    intron_intervals = intervals[intervals["entity"] == INTRON_ENTITY]
    if intron_intervals.empty:
        return genes

    strands = {g.strand for g in genes}
    result: list[Gene] = []
    for strand in strands:
        strand_genes = sorted(
            [g for g in genes if g.strand == strand], key=lambda g: g.start
        )
        merged: list[Gene] = []
        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                nxt = strand_genes[j]
                cur_end = max(e.end for e in current.exons)
                nxt_start = min(e.start for e in nxt.exons)
                gap = nxt_start - cur_end
                if gap < 0 or gap > MAX_MERGE_GAP:
                    break

                gap_intron_prob = float(probs[cur_end:nxt_start, 1].mean())
                if gap_intron_prob > MERGE_GAP_INTRON_PROB:
                    current.exons = sorted(
                        current.exons + nxt.exons, key=lambda e: e.start
                    )
                    j += 1
                else:
                    break
            merged.append(current)
            i = j
        result.extend(merged)

    return result if result else genes


__all__ = [
    "GENIC_THRESHOLD",
    "INTERGENIC_BOOST",
    "MIN_CDS_RUN",
    "VITERBI_ALPHA",
    "_merge_adjacent_cds_viterbi",
    "decode_viterbi",
]
