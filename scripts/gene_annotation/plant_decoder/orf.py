"""ORF correction and intron-probability gene merging.

Ported from PlantHelixSeek's ``pred_to_gff3`` ORF helpers.  Frame shifts are
parameterised: callers pass ``orf_max_shift`` instead of the source module's
mutable ``FRAME_SHIFTS`` global, so decoding never mutates process state.
"""

from __future__ import annotations

import logging

import numpy as np

from .model import Exon, Gene
from .viterbi_utils import INTRON_BILUO_IDX

logger = logging.getLogger(__name__)

STOP_CODONS = {"TAA", "TAG", "TGA"}
MIN_ORF_AA = 15
ATG_WINDOW = 150
STOP_EXTEND_WINDOW = 150
STOP_EXTEND_MIN_PROB = 0.10
MIN_INTRON_LEN = 15
MERGE_INTRON_PROB = 0.60
MERGE_INTRON_MIN_GAP = 50
MIN_CDS_WITH_STOP = 150
ORF_MAX_SHIFT = 3
CDS_IDX_FOR_EXTENSION = slice(1, 5)


def rc(seq: str) -> str:
    return seq.translate(str.maketrans("ATGC", "TACG"))[::-1]


def frame_shifts(orf_max_shift: int = ORF_MAX_SHIFT) -> tuple[int, ...]:
    """Return the symmetric shift range ``(-orf_max_shift, ..., +orf_max_shift)``."""
    if orf_max_shift < 0:
        raise ValueError(f"orf_max_shift must be non-negative; got {orf_max_shift}")
    return tuple(range(-orf_max_shift, orf_max_shift + 1))


def get_cds_seq(exons: list[Exon], seq: str, strand: str) -> str:
    coding = "".join(seq[e.start : e.end] for e in sorted(exons, key=lambda x: x.start))
    return rc(coding) if strand == "-" else coding


def find_longest_orf(
    cds_seq: str, min_aa: int = MIN_ORF_AA
) -> tuple[int, int] | None:
    best_start, best_end = -1, -1
    best_len = 0
    for frame in range(3):
        pos = frame
        while pos + 3 <= len(cds_seq):
            if cds_seq[pos : pos + 3] == "ATG":
                orf_end = pos + 3
                while orf_end + 3 <= len(cds_seq):
                    codon = cds_seq[orf_end : orf_end + 3]
                    if codon in STOP_CODONS:
                        orf_len = orf_end - pos
                        if orf_len >= min_aa * 3 and orf_len > best_len:
                            best_start = pos
                            best_end = orf_end + 3
                            best_len = orf_len
                        break
                    orf_end += 3
                else:
                    orf_len = orf_end - pos
                    remainder = orf_len % 3
                    if remainder:
                        orf_end -= remainder
                        orf_len = orf_end - pos
                    if orf_len >= min_aa * 3 and orf_len > best_len:
                        best_start = pos
                        best_end = orf_end
                        best_len = orf_len
            pos += 3
    if best_start < 0:
        return None
    return (best_start, best_end)


def cds_pos_to_genomic(cds_pos: int, sorted_exons: list[Exon], strand: str) -> int:
    total_cds = sum(e.end - e.start for e in sorted_exons)
    if strand == "-":
        forward_pos = total_cds - 1 - cds_pos
    else:
        forward_pos = cds_pos
    offset = 0
    for exon in sorted_exons:
        exon_len = exon.end - exon.start
        if offset + exon_len > forward_pos:
            return exon.start + (forward_pos - offset)
        offset += exon_len
    return sorted_exons[-1].end


def extract_orf_exons(
    orf_start_cds: int, orf_end_cds: int, exons: list[Exon], strand: str
) -> list[Exon]:
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if strand == "+":
        gen_start = cds_pos_to_genomic(orf_start_cds, sorted_exons, "+")
        gen_end = cds_pos_to_genomic(orf_end_cds - 1, sorted_exons, "+") + 1
    else:
        gen_start = cds_pos_to_genomic(orf_end_cds - 1, sorted_exons, "-")
        gen_end = cds_pos_to_genomic(orf_start_cds, sorted_exons, "-") + 1
    new_exons = []
    for exon in sorted_exons:
        overlap_start = max(exon.start, gen_start)
        overlap_end = min(exon.end, gen_end)
        if overlap_end - overlap_start >= 1:
            new_exons.append(Exon(overlap_start, overlap_end, strand))
    return new_exons


def try_extend_atg(
    cds_seq: str, exons: list[Exon], seq: str, strand: str
) -> tuple[str, list[Exon]]:
    if cds_seq[:3] == "ATG":
        return cds_seq, exons
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if strand == "+":
        first_start = sorted_exons[0].start
        search_start = max(0, first_start - ATG_WINDOW)
        for pos in range(first_start - 3, search_start - 1, -3):
            if pos >= 0 and pos + 3 <= len(seq) and seq[pos : pos + 3] == "ATG":
                new_first = Exon(pos, sorted_exons[0].end, strand)
                new_exons = [new_first] + sorted_exons[1:]
                return get_cds_seq(new_exons, seq, strand), new_exons
    else:
        last_end = sorted_exons[-1].end
        search_end = min(len(seq), last_end + ATG_WINDOW)
        for pos in range(last_end, min(search_end, len(seq) - 2)):
            if rc(seq[pos : pos + 3]) == "ATG":
                new_last = Exon(sorted_exons[-1].start, pos + 3, strand)
                new_exons = sorted_exons[:-1] + [new_last]
                return get_cds_seq(new_exons, seq, strand), new_exons
    return cds_seq, exons


def _try_shifted_exons(
    exons: list[Exon],
    seq: str,
    strand: str,
    frame_shift_range: tuple[int, ...],
) -> tuple[list[Exon], int] | None:
    sorted_exons = sorted(exons, key=lambda e: e.start)
    if len(sorted_exons) < 2:
        return None
    n_pairs = len(sorted_exons) - 1
    trial = list(sorted_exons)
    baseline_cds = get_cds_seq(trial, seq, strand)
    baseline_orf = find_longest_orf(baseline_cds, min_aa=MIN_ORF_AA)
    if baseline_orf is None:
        return None
    best_orf_aa = (baseline_orf[1] - baseline_orf[0]) // 3
    improved = True
    while improved:
        improved = False
        for bi in range(n_pairs):
            best_shift = 0
            best_aa_here = best_orf_aa
            for shift in frame_shift_range:
                if shift == 0:
                    continue
                test = list(trial)
                left = test[bi]
                right = test[bi + 1]
                new_left_end = left.end + shift
                new_right_start = right.start + shift
                if new_left_end <= left.start + 3 or new_right_start >= right.end - 3:
                    continue
                if new_right_start - new_left_end < MIN_INTRON_LEN:
                    continue
                test[bi] = Exon(left.start, new_left_end, strand)
                test[bi + 1] = Exon(new_right_start, right.end, strand)
                cds = get_cds_seq(test, seq, strand)
                orf = find_longest_orf(cds, min_aa=MIN_ORF_AA)
                if orf is None:
                    continue
                if cds[orf[0] : orf[0] + 3] != "ATG":
                    continue
                orf_aa = (orf[1] - orf[0]) // 3
                if orf_aa > best_aa_here:
                    best_aa_here = orf_aa
                    best_shift = shift
            if best_shift != 0:
                left = trial[bi]
                right = trial[bi + 1]
                trial[bi] = Exon(left.start, left.end + best_shift, strand)
                trial[bi + 1] = Exon(right.start + best_shift, right.end, strand)
                best_orf_aa = best_aa_here
                improved = True
    if best_orf_aa > (baseline_orf[1] - baseline_orf[0]) // 3:
        return trial, best_orf_aa
    return None


def correct_gene_orf(
    gene: Gene, seq: str, *, orf_max_shift: int = ORF_MAX_SHIFT
) -> Gene | None:
    sorted_exons = sorted(gene.exons, key=lambda e: e.start)
    n_exons = len(sorted_exons)
    cds_seq = get_cds_seq(sorted_exons, seq, gene.strand)
    cds_seq, sorted_exons = try_extend_atg(cds_seq, sorted_exons, seq, gene.strand)
    orf = find_longest_orf(cds_seq, min_aa=MIN_ORF_AA)
    if n_exons >= 2:
        shifted = _try_shifted_exons(
            sorted_exons, seq, gene.strand, frame_shifts(orf_max_shift)
        )
        if shifted is not None:
            shifted_exons, shifted_orf_aa = shifted
            current_orf_aa = ((orf[1] - orf[0]) // 3) if orf else 0
            if shifted_orf_aa > current_orf_aa:
                sorted_exons = shifted_exons
                cds_seq = get_cds_seq(sorted_exons, seq, gene.strand)
                orf = find_longest_orf(cds_seq, min_aa=MIN_ORF_AA)
    if orf is None:
        return None
    orf_start, orf_end = orf
    if cds_seq[orf_start : orf_start + 3] != "ATG":
        return None
    new_exons = extract_orf_exons(orf_start, orf_end, sorted_exons, gene.strand)
    if not new_exons:
        return None
    total_cds = sum(e.end - e.start for e in new_exons)
    if total_cds < MIN_ORF_AA * 3:
        return None
    final_cds = get_cds_seq(new_exons, seq, gene.strand)
    if final_cds[:3] != "ATG":
        return None
    check_end = len(final_cds)
    if len(final_cds) >= 3 and final_cds[-3:] in STOP_CODONS:
        check_end = len(final_cds) - 3
    has_internal_stop = any(
        final_cds[p : p + 3] in STOP_CODONS for p in range(0, max(0, check_end - 2), 3)
    )
    if has_internal_stop:
        return None
    total = sum(e.end - e.start for e in new_exons)
    rem = total % 3
    if rem and gene.strand == "+":
        last = new_exons[-1]
        if last.end - last.start > rem:
            new_exons[-1] = Exon(last.start, last.end - rem, gene.strand)
    elif rem and gene.strand == "-":
        first = new_exons[0]
        if first.end - first.start > rem:
            new_exons[0] = Exon(first.start + rem, first.end, gene.strand)
    gene.exons = new_exons
    return gene


def _merge_gene_short_introns(gene: Gene) -> Gene:
    exs = sorted(gene.exons, key=lambda e: e.start)
    if len(exs) <= 1:
        return gene
    filtered = [exs[0]]
    for i in range(1, len(exs)):
        gap = exs[i].start - filtered[-1].end
        if gap < MIN_INTRON_LEN:
            prev_len = filtered[-1].end - filtered[-1].start
            curr_len = exs[i].end - exs[i].start
            if curr_len > prev_len:
                filtered[-1] = exs[i]
        else:
            filtered.append(exs[i])
    gene.exons = filtered
    return gene


def try_extend_stop(gene: Gene, seq: str, probs: np.ndarray) -> bool:
    if not gene.exons:
        return False
    cds_seq = get_cds_seq(gene.exons, seq, gene.strand)
    if len(cds_seq) >= 3 and cds_seq[-3:] in STOP_CODONS:
        return False
    sorted_exons = sorted(gene.exons, key=lambda e: e.start)
    if gene.strand == "+":
        search_start = sorted_exons[-1].end
        search_end = min(len(seq), search_start + STOP_EXTEND_WINDOW)
        for pos in range(search_start, search_end - 2, 3):
            if seq[pos : pos + 3] in STOP_CODONS:
                ext_cds_prob = float(
                    probs[search_start : pos + 3, CDS_IDX_FOR_EXTENSION].sum(axis=1).mean()
                )
                if ext_cds_prob >= STOP_EXTEND_MIN_PROB:
                    sorted_exons[-1] = Exon(
                        sorted_exons[-1].start, pos + 3, gene.strand
                    )
                    gene.exons = sorted_exons
                    return True
                break
    else:
        search_end = sorted_exons[0].start
        search_start = max(0, search_end - STOP_EXTEND_WINDOW)
        for pos in range(search_end - 3, search_start - 1, -3):
            if rc(seq[pos : pos + 3]) in STOP_CODONS:
                ext_cds_prob = float(
                    probs[pos:search_end, CDS_IDX_FOR_EXTENSION].sum(axis=1).mean()
                )
                if ext_cds_prob >= STOP_EXTEND_MIN_PROB:
                    sorted_exons[0] = Exon(pos, sorted_exons[0].end, gene.strand)
                    gene.exons = sorted_exons
                    return True
                break
    return False


def correct_all_genes(
    genes: list[Gene],
    seq: str,
    fwd_probs: np.ndarray | None = None,
    rev_probs: np.ndarray | None = None,
    *,
    orf_max_shift: int = ORF_MAX_SHIFT,
) -> list[Gene]:
    kept = 0
    dropped = 0
    corrected = []
    for gene in genes:
        result = correct_gene_orf(gene, seq, orf_max_shift=orf_max_shift)
        if result is not None:
            corrected.append(result)
            kept += 1
        else:
            dropped += 1
    revalidated = []
    for gene in corrected:
        exs = sorted(gene.exons, key=lambda e: e.start)
        has_short = any(
            exs[i + 1].start - exs[i].end < MIN_INTRON_LEN
            for i in range(len(exs) - 1)
        )
        if has_short:
            gene = _merge_gene_short_introns(gene)
            result = correct_gene_orf(gene, seq, orf_max_shift=orf_max_shift)
            if result is not None:
                revalidated.append(result)
            else:
                revalidated.append(gene)
        else:
            revalidated.append(gene)
    if fwd_probs is not None and rev_probs is not None:
        extended = 0
        for gene in revalidated:
            probs = fwd_probs if gene.strand == "+" else rev_probs
            if try_extend_stop(gene, seq, probs):
                extended += 1
        if extended:
            logger.info("Stop codon extension: %d genes extended", extended)
    before = len(revalidated)
    revalidated = [
        g
        for g in revalidated
        if (sum(e.end - e.start for e in g.exons) >= MIN_CDS_WITH_STOP)
        or (
            len(get_cds_seq(g.exons, seq, g.strand)) >= 3
            and get_cds_seq(g.exons, seq, g.strand)[-3:] in STOP_CODONS
        )
    ]
    if len(revalidated) < before:
        logger.info(
            "Stop codon filter: %d removed, %d kept",
            before - len(revalidated),
            len(revalidated),
        )
    logger.info("ORF correction: %d kept, %d dropped", kept, dropped)
    return revalidated


def merge_genes_by_intron_prob(
    genes: list[Gene], fwd_probs: np.ndarray, rev_probs: np.ndarray
) -> list[Gene]:
    if not genes:
        return genes
    merged_count = 0
    genes_sorted = sorted(genes, key=lambda g: g.start)
    result: list[Gene] = []
    for strand in ("+", "-"):
        strand_genes = [g for g in genes_sorted if g.strand == strand]
        probs = fwd_probs if strand == "+" else rev_probs
        i = 0
        while i < len(strand_genes):
            current = strand_genes[i]
            j = i + 1
            while j < len(strand_genes):
                next_gene = strand_genes[j]
                cur_exons = sorted(current.exons, key=lambda e: e.start)
                nxt_exons = sorted(next_gene.exons, key=lambda e: e.start)
                gap_start = cur_exons[-1].end
                gap_end = nxt_exons[0].start
                gap_len = gap_end - gap_start
                if gap_len < MERGE_INTRON_MIN_GAP:
                    break
                gap_intron_prob = float(
                    probs[gap_start:gap_end, INTRON_BILUO_IDX].sum(axis=1).mean()
                )
                if gap_intron_prob >= MERGE_INTRON_PROB:
                    current.exons = sorted(
                        current.exons + next_gene.exons, key=lambda e: e.start
                    )
                    merged_count += 1
                    j += 1
                else:
                    break
            result.append(current)
            i = j
    if merged_count:
        logger.info("Gene merge by intron prob: %d pairs merged", merged_count)
    return result


__all__ = [
    "ORF_MAX_SHIFT",
    "correct_all_genes",
    "correct_gene_orf",
    "extract_orf_exons",
    "find_longest_orf",
    "frame_shifts",
    "get_cds_seq",
    "merge_genes_by_intron_prob",
    "rc",
    "try_extend_atg",
    "try_extend_stop",
]
