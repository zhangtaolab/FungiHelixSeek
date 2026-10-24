#!/usr/bin/env python3
"""Gap-capped padding of gene/mRNA footprints in a GFF3 file."""

from __future__ import annotations

from pathlib import Path


def _attrs(field: str) -> dict[str, str]:
    out = {}
    for part in field.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def _parse(line: str):
    fields = line.split("\t")
    return fields if len(fields) == 9 else None


def pad_gene_spans(gff3: str, pad: int, out: str) -> None:
    """Extend gene and mRNA start/end by ``pad`` bp without crossing the
    midpoint of the intergenic gap to a neighbouring gene."""
    rows = Path(gff3).read_text().splitlines()
    headers = [r for r in rows if r.startswith("#")]
    body = [r for r in rows if r and not r.startswith("#")]

    by_seq: dict[str, list[tuple[int, int, int]]] = {}
    for i, line in enumerate(body):
        fields = _parse(line)
        if fields is None or fields[2] != "gene":
            continue
        by_seq.setdefault(fields[0], []).append(
            (i, int(fields[3]), int(fields[4])))

    new_span: dict[tuple[int, str], tuple[int, int]] = {}
    for seqid, items in by_seq.items():
        items.sort(key=lambda x: x[1])
        for k, (idx, start, end) in enumerate(items):
            new_start = max(1, start - pad)
            new_end = end + pad
            if k > 0:
                new_start = max(new_start, (items[k - 1][2] + start) // 2 + 1)
            if k + 1 < len(items):
                new_end = min(new_end, (end + items[k + 1][1]) // 2)
            new_start = min(new_start, start)
            new_end = max(new_end, end)
            gid = _attrs(_parse(body[idx])[8]).get("ID", "")
            new_span[(idx, gid)] = (new_start, new_end)

    gene_span = {gid: span for (_idx, gid), span in new_span.items()}
    rewritten: list[str] = []
    for i, line in enumerate(body):
        fields = _parse(line)
        if fields is None or fields[2] not in ("gene", "mRNA"):
            rewritten.append(line)
            continue
        attrs = _attrs(fields[8])
        gid = attrs.get("ID", "") if fields[2] == "gene" else \
            attrs.get("Parent", "").split(",")[0]
        span = gene_span.get(gid)
        if span is None:
            rewritten.append(line)
            continue
        fields[3], fields[4] = str(span[0]), str(span[1])
        rewritten.append("\t".join(fields))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for line in headers:
            fh.write(line + "\n")
        for line in rewritten:
            fh.write(line + "\n")


__all__ = ["pad_gene_spans"]
