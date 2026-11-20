"""Native GFF3 writer for the rich :class:`plant_decoder.model.Gene` model.

Decoder coordinates are half-open 0-based ``[start, end)``; GFF3 is 1-based
inclusive, so this boundary converts ``[start, end)`` to ``start + 1 .. end``.
Feature schema and ``Parent=`` attributes match the existing Fungi GFF output.
Exons are the union of CDS and UTR spans, merging only overlapping or directly
adjacent intervals.
"""

from __future__ import annotations

from pathlib import Path

from .model import Exon, Gene

Segment = tuple[int, int]

__all__ = ["gene_to_gff_lines", "write_gff3"]


def _spans(exons: list[Exon]) -> list[Segment]:
    """Return sorted, valid half-open spans from a list of ``Exon`` objects."""
    return sorted(
        (e.start, e.end) for e in exons if e.end > e.start
    )


def _merge_intervals(spans: list[Segment]) -> list[Segment]:
    """Merge overlapping or directly adjacent half-open intervals.

    ``[a, b)`` and ``[b, c)`` are directly adjacent and merge into ``[a, c)``.
    Intervals separated by at least one base remain distinct.
    """
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _cds_phases(cds_spans: list[Segment], strand: str) -> list[int]:
    """GFF3 phase per CDS span, aligned to ``cds_spans``, in transcription order.

    Each span is tracked by its list index so duplicate coordinates still get
    sequential phases rather than sharing one dictionary key.
    """
    phases = [0] * len(cds_spans)
    ordered_indices = sorted(
        range(len(cds_spans)),
        key=lambda index: cds_spans[index],
        reverse=(strand == "-"),
    )
    cumulative = 0
    for index in ordered_indices:
        start, end = cds_spans[index]
        phases[index] = (3 - cumulative % 3) % 3
        cumulative += end - start
    return phases


def gene_to_gff_lines(
    seqid: str,
    gene: Gene,
    gid: str,
    source: str,
) -> list[str]:
    """Render one ``Gene`` as GFF3 lines using the supplied ``gid``.

    ``gid`` is used verbatim: gene identity is ``ID=<gid>`` and the single
    transcript identity is ``ID=<gid>.t1;Parent=<gid>``.  Gene/mRNA spans cover
    the min/max of all CDS and UTR spans.  CDS and UTR features keep their
    individual spans; exons are the union of those spans.  Rows within each
    section are sorted deterministically by coordinate (and feature type).
    """
    strand = gene.strand
    cds = _spans(gene.exons)
    utr5 = _spans(gene.utr5)
    utr3 = _spans(gene.utr3)
    all_spans = cds + utr5 + utr3
    if not all_spans:
        return []

    gene_start = min(start for start, _ in all_spans) + 1
    gene_end = max(end for _, end in all_spans)
    rna_id = f"{gid}.t1"

    lines: list[str] = [
        f"{seqid}\t{source}\tgene\t{gene_start}\t{gene_end}\t.\t{strand}\t.\tID={gid}",
        f"{seqid}\t{source}\tmRNA\t{gene_start}\t{gene_end}\t.\t{strand}\t.\t"
        f"ID={rna_id};Parent={gid}",
    ]

    for start, end in _merge_intervals(all_spans):
        lines.append(
            f"{seqid}\t{source}\texon\t{start + 1}\t{end}\t.\t{strand}\t.\t"
            f"Parent={rna_id}"
        )

    phases = _cds_phases(cds, strand)
    features: list[tuple[str, int, int, str]] = [
        ("CDS", start, end, str(phases[index]))
        for index, (start, end) in enumerate(cds)
    ]
    features.extend(
        ("five_prime_UTR", start, end, ".") for start, end in utr5
    )
    features.extend(
        ("three_prime_UTR", start, end, ".") for start, end in utr3
    )
    features.sort(key=lambda feature: (feature[1], feature[2], feature[0]))

    for feature_type, start, end, phase in features:
        lines.append(
            f"{seqid}\t{source}\t{feature_type}\t{start + 1}\t{end}\t.\t"
            f"{strand}\t{phase}\tParent={rna_id}"
        )

    return lines


def _has_spans(gene: Gene) -> bool:
    return bool(_spans(gene.exons) or _spans(gene.utr5) or _spans(gene.utr3))


def _ordered_genes(genes: list[Gene]) -> list[Gene]:
    """Order genes by contig first appearance, then by genomic coordinate.

    Contig first-appearance order preserves FASTA/global ordering when callers
    pass genes contig-by-contig; within a contig genes are sorted by start/end
    so output is deterministic regardless of input order.
    """
    groups: dict[str, list[Gene]] = {}
    contig_order: list[str] = []
    for gene in genes:
        if gene.chrom not in groups:
            groups[gene.chrom] = []
            contig_order.append(gene.chrom)
        groups[gene.chrom].append(gene)

    ordered: list[Gene] = []
    for chrom in contig_order:
        ordered.extend(
            sorted(groups[chrom], key=lambda g: (g.start, g.end, g.gid))
        )
    return ordered


def write_gff3(
    genes: list[Gene],
    path: str | Path,
    source: str,
    gid_prefix: str | None = None,
    start_index: int = 1,
) -> None:
    """Write ``genes`` to ``path`` as a GFF3 file.

    By default IDs follow the Fungi convention ``<source>_<contig>_g<N>`` with a
    single global counter starting at ``start_index`` (FASTA/global numbering).
    When ``gid_prefix`` is supplied it replaces the ``<source>_<contig>_g``
    prefix verbatim and the running integer is appended directly, e.g.
    ``gid_prefix="run_g"`` produces ``run_g7`` for ``start_index=7``.
    """
    out_path = Path(path)
    ordered = _ordered_genes([gene for gene in genes if _has_spans(gene)])

    lines = ["##gff-version 3"]
    number = start_index
    for gene in ordered:
        if gid_prefix is None:
            prefix = f"{source}_{gene.chrom}_g"
        else:
            prefix = str(gid_prefix)
        gid = f"{prefix}{number}"
        lines.extend(gene_to_gff_lines(gene.chrom, gene, gid, source))
        number += 1

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
