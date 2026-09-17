def gene_to_gff_lines(seqid, gene, gid, source="FungiHelixSeek", strand="+"):
    L = []
    a = L.append
    span_starts = [gene["start"]] + [gene["cds"][0][0]]
    span_ends = [gene["end"]] + [gene["cds"][-1][1]]
    for u in (gene["u5"], gene["u3"]):
        if u:
            span_starts.append(u[0])
            span_ends.append(u[1])
    gstart = min(span_starts) + 1
    gend = max(span_ends) + 1
    a(f"{seqid}\t{source}\tgene\t{gstart}\t{gend}\t.\t{strand}\t.\tID={gid}")
    a(f"{seqid}\t{source}\tmRNA\t{gstart}\t{gend}\t.\t{strand}\t.\t"
      f"ID={gid}.t1;Parent={gid}")

    segments = []
    if gene["u5"]:
        segments.append(("five_prime_UTR", gene["u5"][0], gene["u5"][1]))
    cds = sorted(gene["cds"])
    introns = sorted(gene["introns"])
    cds_phase_map = {}
    segs_trans = sorted(cds, key=lambda x: x[0], reverse=(strand == "-"))
    acc = 0
    for s, e in segs_trans:
        cds_phase_map[(s, e)] = (3 - acc % 3) % 3
        acc += e - s + 1
    for i, (s, e) in enumerate(cds):
        segments.append(("CDS", s, e))
        if i < len(introns):
            segments.append(("intron", introns[i][0], introns[i][1]))
    if gene["u3"]:
        segments.append(("three_prime_UTR", gene["u3"][0], gene["u3"][1]))

    segments.sort(key=lambda x: x[1])
    exon_intervals = []
    cur = None
    for ftype, s, e in segments:
        if ftype == "intron":
            if cur:
                exon_intervals.append(cur)
                cur = None
        else:
            cur = (cur[0], e) if cur else (s, e)
    if cur:
        exon_intervals.append(cur)

    for s, e in exon_intervals:
        a(f"{seqid}\t{source}\texon\t{s + 1}\t{e + 1}\t.\t{strand}\t.\tParent={gid}.t1")

    for ftype, s, e in segments:
        if ftype == "intron":
            continue
        phase = str(cds_phase_map[(s, e)]) if ftype == "CDS" else "."
        a(f"{seqid}\t{source}\t{ftype}\t{s + 1}\t{e + 1}\t.\t{strand}\t{phase}\tParent={gid}.t1")
    return L


def cds_phases(cds_segments, strand="+"):
    phases = []
    acc = 0
    segs = sorted(cds_segments, key=lambda x: x[0], reverse=(strand == "-"))
    for s, e in segs:
        phases.append((3 - acc % 3) % 3)
        acc += e - s + 1
    return phases
