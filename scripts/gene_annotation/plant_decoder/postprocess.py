"""UTR detection and PlantHelixSeek-style post-processing filters.

Ported from PlantHelixSeek's ``pred_to_gff3`` helpers.  Functions operate on
half-open ``Exon`` spans and mutate only the genes passed in.
"""

from __future__ import annotations

import logging

import numpy as np

from .model import Exon, Gene
from .viterbi_utils import UTR3_BILUO_IDX, UTR5_BILUO_IDX

logger = logging.getLogger(__name__)

UTR_SEARCH_WINDOW = 3000
MIN_CDS_FEATURE = 6
MIN_UTR_FEATURE = 3
MIN_GENE_LEN = 30
SINGLE_EXON_MIN_PROB = 0.7
SINGLE_EXON_MIN_CDS = 200


def _extract_utr_spans(
    labels: np.ndarray, region_start: int, region_end: int, utr_indices: list[int]
) -> list[tuple[int, int]]:
    spans = []
    pos = region_start
    while pos < region_end:
        if labels[pos] in utr_indices:
            s = pos
            while pos < region_end and labels[pos] in utr_indices:
                pos += 1
            if pos - s >= MIN_UTR_FEATURE:
                spans.append((s, pos))
        else:
            pos += 1
    return spans


def detect_all_utrs(
    genes: list[Gene], fwd_labels: np.ndarray, rev_labels: np.ndarray
) -> None:
    occupied_fwd = set()
    occupied_rev = set()
    for g in genes:
        occ = occupied_fwd if g.strand == "+" else occupied_rev
        for e in g.exons:
            occ.update(range(e.start, e.end))
    fwd_genes = sorted(
        [g for g in genes if g.strand == "+"], key=lambda g: g.exons[0].start
    )
    rev_genes = sorted(
        [g for g in genes if g.strand == "-"], key=lambda g: g.exons[0].start
    )

    def _gene_boundaries(sorted_genes):
        return [
            (
                g,
                sorted(g.exons, key=lambda e: e.start)[0].start,
                sorted(g.exons, key=lambda e: e.start)[-1].end,
            )
            for g in sorted_genes
        ]

    def _clip_search(
        search_start, search_end, gene_cds_start, gene_cds_end, boundaries, gene_idx
    ):
        for i, (_, other_start, other_end) in enumerate(boundaries):
            if i == gene_idx:
                continue
            if other_start < gene_cds_start and other_end > search_start:
                search_start = max(search_start, other_end)
            if other_end > gene_cds_end and other_start < search_end:
                search_end = min(search_end, other_start)
        return search_start, search_end

    for gene_list in [fwd_genes, rev_genes]:
        boundaries = _gene_boundaries(gene_list)
        for idx, (g, _, _) in enumerate(boundaries):
            labels = fwd_labels if g.strand == "+" else rev_labels
            occupied = occupied_fwd if g.strand == "+" else occupied_rev
            exs = sorted(g.exons, key=lambda e: e.start)
            first_cds_start = exs[0].start
            last_cds_end = exs[-1].end
            if g.strand == "+":
                us = max(0, first_cds_start - UTR_SEARCH_WINDOW)
                ds = min(len(labels), last_cds_end + UTR_SEARCH_WINDOW)
                us, _ = _clip_search(
                    us, first_cds_start, first_cds_start, last_cds_end, boundaries, idx
                )
                _, ds = _clip_search(
                    last_cds_end, ds, first_cds_start, last_cds_end, boundaries, idx
                )
                raw5 = _extract_utr_spans(labels, us, first_cds_start, UTR5_BILUO_IDX)
                raw3 = _extract_utr_spans(labels, last_cds_end, ds, UTR3_BILUO_IDX)
            else:
                us = min(len(labels), last_cds_end + UTR_SEARCH_WINDOW)
                ds = max(0, first_cds_start - UTR_SEARCH_WINDOW)
                _, us = _clip_search(
                    last_cds_end, us, first_cds_start, last_cds_end, boundaries, idx
                )
                ds, _ = _clip_search(
                    ds, first_cds_start, first_cds_start, last_cds_end, boundaries, idx
                )
                raw5 = _extract_utr_spans(labels, last_cds_end, us, UTR5_BILUO_IDX)
                raw3 = _extract_utr_spans(labels, ds, first_cds_start, UTR3_BILUO_IDX)
            g.utr5 = [
                Exon(s, e, g.strand)
                for s, e in raw5
                if not any(p in occupied for p in range(s, e))
            ]
            g.utr3 = [
                Exon(s, e, g.strand)
                for s, e in raw3
                if not any(p in occupied for p in range(s, e))
            ]


def filter_to_min_cds_length(
    genes: list[Gene], min_len: int = MIN_CDS_FEATURE
) -> list[Gene]:
    before = len(genes)
    kept = [
        g for g in genes if sum(e.end - e.start for e in g.exons) >= min_len
    ]
    removed = before - len(kept)
    if removed:
        logger.info("Min CDS filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_to_min_gene_length(
    genes: list[Gene], min_len: int = MIN_GENE_LEN
) -> list[Gene]:
    before = len(genes)
    kept = [
        g
        for g in genes
        if (max(e.end for e in g.exons) - min(e.start for e in g.exons)) >= min_len
    ]
    removed = before - len(kept)
    if removed:
        logger.info("Min gene length filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_to_valid_genes(
    genes: list[Gene], require_utrs: bool = False
) -> list[Gene]:
    before = len(genes)
    kept = [g for g in genes if g.exons]
    removed = before - len(kept)
    if removed:
        logger.info("Valid gene filter: %d removed, %d kept", removed, len(kept))
    return kept


def filter_low_confidence_single_exons(
    genes: list[Gene],
    fwd_probs: np.ndarray,
    rev_probs: np.ndarray,
    min_prob: float = SINGLE_EXON_MIN_PROB,
    min_cds_len: int = SINGLE_EXON_MIN_CDS,
) -> list[Gene]:
    """Remove single-exon genes below CDS-probability or CDS-length thresholds.

    A single-exon gene is removed if either its mean 17-class CDS probability is
    below ``min_prob`` or its total CDS length is below ``min_cds_len``.
    Multi-exon genes are always kept.
    """
    if min_prob <= 0 and min_cds_len <= 0:
        return genes
    cds_idx = slice(1, 5)
    kept = []
    removed = 0
    for g in genes:
        if len(g.exons) >= 2:
            kept.append(g)
            continue
        cds_len = sum(e.end - e.start for e in g.exons)
        if min_cds_len > 0 and cds_len < min_cds_len:
            removed += 1
            continue
        if min_prob > 0:
            probs = fwd_probs if g.strand == "+" else rev_probs
            ex = g.exons[0]
            mean_cds = float(probs[ex.start : ex.end, cds_idx].sum(axis=1).mean())
            if mean_cds < min_prob:
                removed += 1
                continue
        kept.append(g)
    if removed:
        logger.info(
            "Single-exon filter (prob>= %.2f, len>= %d): %d removed, %d kept",
            min_prob,
            min_cds_len,
            removed,
            len(kept),
        )
    return kept


__all__ = [
    "MIN_CDS_FEATURE",
    "MIN_GENE_LEN",
    "MIN_UTR_FEATURE",
    "SINGLE_EXON_MIN_CDS",
    "SINGLE_EXON_MIN_PROB",
    "UTR_SEARCH_WINDOW",
    "detect_all_utrs",
    "filter_low_confidence_single_exons",
    "filter_to_min_cds_length",
    "filter_to_min_gene_length",
    "filter_to_valid_genes",
]
