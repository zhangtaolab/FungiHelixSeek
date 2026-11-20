"""Gene model dataclasses shared by the vendored Plant decoder.

Coordinates are **half-open** genomic intervals ``[start, end)`` as used by
PlantHelixSeek.  The public Fungi API/GFF writer converts to 1-based inclusive
GFF3 coordinates at its boundary; internal decoding stays half-open.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Exon:
    """A half-open genomic interval ``[start, end)`` on ``strand``."""

    start: int
    end: int
    strand: str


@dataclass
class Gene:
    """A gene model with CDS exons and optional 5'/3' UTR spans.

    All intervals are half-open.  ``exons`` holds CDS spans only; UTR spans are
    kept in ``utr5`` / ``utr3`` so multi-span UTRs are preserved.
    """

    gid: str
    chrom: str
    strand: str
    exons: list[Exon] = field(default_factory=list)
    utr5: list[Exon] = field(default_factory=list)
    utr3: list[Exon] = field(default_factory=list)

    @property
    def start(self):
        all_s = [e.start for e in self.exons] + [e.start for e in self.utr5] + [e.start for e in self.utr3]
        return min(all_s) if all_s else min(e.start for e in self.exons)

    @property
    def end(self):
        all_e = [e.end for e in self.exons] + [e.end for e in self.utr5] + [e.end for e in self.utr3]
        return max(all_e) if all_e else max(e.end for e in self.exons)

    @property
    def cds_len(self):
        return sum(e.end - e.start for e in self.exons)
