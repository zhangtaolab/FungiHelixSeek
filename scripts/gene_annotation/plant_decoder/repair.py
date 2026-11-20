"""Boundary-anchored terminal ORF repair for gene models.

Faithful port of PlantHelixSeek's ``terminal_repair`` ORF helpers.  Internal
splice sites are kept fixed; only the first and last CDS boundaries are moved
to obtain a valid ORF (ATG, in-frame stop, length divisible by 3, no internal
stop).  If no valid repair is found within the search windows the gene is left
unchanged.  Splice fixing / alternative-ORF logic is intentionally *not* part
of this module.

Coordinates are half-open genomic ``[start, end)`` intervals as used by the
:class:`~plant_decoder.model.Exon`/:class:`~plant_decoder.model.Gene` model.
Minus-strand genes are repaired on the reverse-complement chromosome.
"""

from __future__ import annotations

from .model import Exon, Gene

STOP = {"TAA", "TAG", "TGA"}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

DEFAULT_W5 = 300
DEFAULT_W3 = 300


def rc(s: str) -> str:
    """Reverse-complement ``s`` (DNA, IUPAC ``N`` preserved)."""
    return s.translate(COMP)[::-1]


def cds_seq(exons: list[tuple[int, int]], seq: str) -> str:
    """Concatenate the spliced coding sequence over ``exons``."""
    return "".join(seq[s:e] for s, e in sorted(exons))


def first_stop(cds: str) -> int | None:
    """Return the end position of the first in-frame stop codon, or ``None``."""
    for q in range(0, len(cds) - 2, 3):
        if cds[q : q + 3] in STOP:
            return q + 3
    return None


def valid_orf(exons: list[tuple[int, int]], seq: str) -> bool:
    """Whether ``exons`` form a complete valid ORF (ATG..stop, no internal stop)."""
    if not exons:
        return False
    c = cds_seq(exons, seq)
    if len(c) < 60 or len(c) % 3 != 0:
        return False
    if c[:3] != "ATG" or c[-3:] not in STOP:
        return False
    return all(c[q : q + 3] not in STOP for q in range(0, len(c) - 3, 3))


def _map_cds_to_exon(
    cds_pos: int, exons: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Map a spliced CDS position to ``(exon_index, offset_within_exon)``."""
    remaining = cds_pos
    for i, (s, e) in enumerate(exons):
        length = e - s
        if remaining <= length:
            return i, remaining
        remaining -= length
    return None


def repair_exons(
    exons: list[tuple[int, int]],
    seq: str,
    w5: int = DEFAULT_W5,
    w3: int = DEFAULT_W3,
    start_rule: str = "nearest",
) -> list[tuple[int, int]] | None:
    """Repair a plus-orientation CDS chain by moving only terminal boundaries.

    Returns the repaired exon list (plus orientation) or ``None`` when no valid
    terminal repair exists within ``w5``/``w3``.
    """
    if not exons:
        return None
    exons = sorted(exons)
    pred5, pred3 = exons[0][0], exons[-1][1]
    if pred3 - pred5 < 60:
        return None

    lo = max(0, pred5 - w5)
    hi = min(exons[0][1] - 3, len(seq) - 3)
    starts = [p for p in range(lo, hi + 1) if seq[p : p + 3] == "ATG"]
    if not starts:
        return None

    candidates: list[tuple[tuple, list[tuple[int, int]]]] = []
    for p in starts:
        test_exons = [(p, exons[0][1])] + exons[1:]
        cds = cds_seq(test_exons, seq)
        stop_end = first_stop(cds)
        if stop_end is None:
            continue
        mapped = _map_cds_to_exon(stop_end, test_exons)
        if mapped is None:
            continue
        idx, off = mapped
        if idx != len(test_exons) - 1:
            continue
        stop_genomic = test_exons[-1][0] + off
        if stop_genomic <= test_exons[-1][0] + 3:
            continue
        if abs(stop_genomic - pred3) > w3:
            continue
        final = [(p, exons[0][1])] + exons[1:-1] + [(exons[-1][0], stop_genomic)]
        if not valid_orf(final, seq):
            continue
        dist = abs(p - pred5) + abs(stop_genomic - pred3)
        kozak = 0
        if p >= 3 and seq[p - 3] in "AG":
            kozak -= 4
        if p + 3 < len(seq) and seq[p + 3] == "G":
            kozak -= 2
        if start_rule == "first":
            key = (p, dist)
        elif start_rule == "longest":
            key = (-len(cds_seq(final, seq)), dist)
        else:
            key = (dist + kozak, abs(p - pred5))
        candidates.append((key, final))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def repair_genes(
    genes: list[Gene],
    seq: str,
    rc_seq: str | None = None,
    w5: int = DEFAULT_W5,
    w3: int = DEFAULT_W3,
    start_rule: str = "nearest",
) -> int:
    """Repair a list of :class:`Gene` objects in place.

    Genes that already contain a valid ORF are left untouched.  Returns the
    number of genes actually repaired.
    """
    if rc_seq is None:
        rc_seq = rc(seq)
    length = len(seq)
    n_repaired = 0
    for g in genes:
        if not g.exons:
            continue
        if g.strand == "-":
            work = sorted((length - e.end, length - e.start) for e in g.exons)
            wseq = rc_seq
        else:
            work = sorted((e.start, e.end) for e in g.exons)
            wseq = seq
        if valid_orf(work, wseq):
            continue
        fixed = repair_exons(work, wseq, w5, w3, start_rule)
        if fixed is None:
            continue
        n_repaired += 1
        if g.strand == "-":
            g.exons = [Exon(length - b, length - a, "-") for a, b in fixed]
        else:
            g.exons = [Exon(a, b, "+") for a, b in fixed]
    return n_repaired


__all__ = [
    "COMP",
    "DEFAULT_W3",
    "DEFAULT_W5",
    "STOP",
    "_map_cds_to_exon",
    "cds_seq",
    "first_stop",
    "rc",
    "repair_exons",
    "repair_genes",
    "valid_orf",
]
