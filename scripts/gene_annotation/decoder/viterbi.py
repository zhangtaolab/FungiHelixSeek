import numpy as np

from . import states as ST

NEG = -1e9
TRANS_LOG = -2.0
BND_BONUS = 1.0
EPS = 1e-8
BND_MIX = 0.2
INT_MIX = 0.4

INTRON_MIN, INTRON_MAX = 40, 2000
EXON_MIN = 20
LEN_PENALTY = -6.0

_LEN_TABLE = None


def _length_tables():
    """Empirical log-pdf penalties (peak-normalized) from 60 train species."""
    global _LEN_TABLE
    if _LEN_TABLE is None:
        from pathlib import Path
        path = Path(__file__).resolve().with_name("length_priors.npz")
        if path.exists():
            z = np.load(path)
            _LEN_TABLE = (z["intron"], z["exon"])
        else:
            _LEN_TABLE = False
    return _LEN_TABLE


def _len_penalty(d, table):
    return float(table[min(d, len(table) - 1)])

_CDS_GROUP = frozenset((ST.S0, ST.S1, ST.S2, *ST.CDS,
                        ST.STOP0, ST.STOP1, ST.STOP2))
_INT_GROUP = frozenset(ST.INT)

# p17 column indices (BILUO label ids)
_O, _B_CDS, _I_CDS, _L_CDS = 0, 1, 2, 3
_B_INT, _I_INT, _L_INT = 5, 6, 7
_UTR5 = slice(9, 13)
_UTR3 = slice(13, 17)

_SINGLE_GENE_STATES = frozenset((ST.S0, ST.S1, ST.S2, *ST.CDS, *ST.INT,
                                 ST.STOP0, ST.STOP1, ST.STOP2))


def _group(s):
    if s in _INT_GROUP:
        return 1
    if s in _CDS_GROUP:
        return 2
    return 0


def _precompute_seq_flags(seq):
    L = len(seq)
    is_A = np.frombuffer(seq.encode(), dtype=np.uint8) == ord("A")
    is_T = np.frombuffer(seq.encode(), dtype=np.uint8) == ord("T")
    is_G = np.frombuffer(seq.encode(), dtype=np.uint8) == ord("G")

    atg = np.zeros(L, dtype=bool)
    stop_start = np.zeros(L, dtype=bool)
    donor_ok = np.zeros(L, dtype=bool)
    acceptor_ok = np.zeros(L, dtype=bool)
    for p in range(L - 2):
        tri = seq[p:p + 3]
        if tri == "ATG":
            atg[p] = True
        if tri in ST.STOP_CODONS:
            stop_start[p] = True
    for p in range(L - 1):
        if seq[p:p + 2] in ST.DONORS:
            donor_ok[p] = True
        if seq[p:p + 2] == ST.ACCEPTOR:
            acceptor_ok[p] = True
    return is_A, is_T, is_G, atg, stop_start, donor_ok, acceptor_ok


def build_emissions(seq, p17, p_phase, p_bnd):
    """Per-state static emissions.

    Interiors use coarse class probabilities (sum of B/I/L/U per structure);
    BILUO boundary evidence enters only at the gene-terminal states S0
    (B-CDS) and STOP2 (L-CDS), blended with the coarse floor by BND_MIX.
    Splice-junction edges use the legacy p_bnd aggregate bonus in the DP.
    """
    L = len(seq)
    is_A, is_T, is_G, atg, stop_start, donor_ok, acceptor_ok = \
        _precompute_seq_flags(seq)
    em = np.full((L, ST.N_STATES), NEG, dtype=np.float64)

    p17 = p17.astype(np.float64)
    logp = np.log(p17 + EPS)
    log_phase = np.log(p_phase.astype(np.float64) + EPS)

    em[:, ST.IG] = logp[:, _O]
    em[:, ST.U5] = np.log(p17[:, _UTR5].sum(axis=1) + EPS)
    em[:, ST.U3] = np.log(p17[:, _UTR3].sum(axis=1) + EPS)

    cds_coarse = np.log(p17[:, _B_CDS:_L_CDS + 2].sum(axis=1) + EPS)
    em[:, ST.S0] = np.where(is_A, logp[:, _B_CDS], NEG)
    em[is_T, ST.S1] = cds_coarse[is_T]
    em[is_G, ST.S2] = cds_coarse[is_G]

    for i in range(3):
        cds_em = cds_coarse + log_phase[:, i + 1]
        if i == 0:
            cds_em = cds_em.copy()
            cds_em[stop_start] = NEG
        em[:, ST.CDS[i]] = cds_em

    for i in range(3):
        em[:, ST.INT[i]] = np.log(p17[:, _B_INT:_L_INT + 2].sum(axis=1) + EPS)

    em[stop_start, ST.STOP0] = cds_coarse[stop_start]
    em[:, ST.STOP1] = cds_coarse
    em[:, ST.STOP2] = logp[:, _L_CDS]

    logs = {"B_cds": logp[:, _B_CDS], "L_cds": logp[:, _L_CDS],
            "B_int": logp[:, _B_INT], "L_int": logp[:, _L_INT],
            "phase": log_phase,
            "cds_coarse": cds_coarse,
            "int_coarse": em[:, ST.INT[0]]}
    return em, donor_ok, acceptor_ok, logs


def _edge_emission(src, dst, p, em, logs):
    """Emission for splice-junction edges.

    CDS-side boundary evidence (L-CDS before donor, B-CDS at acceptor) is
    used hard -- it drove the cds_F1 gains. Intron-side evidence (B-INTRON
    at donor, L-INTRON before acceptor) is blended with a coarse floor
    (INT_MIX) -- used hard it cost intron recall (boundary tags occasionally
    sit 1-2bp off the GT-AG site the hard sequence constraint then forbids).
    """
    if dst in _INT_GROUP and src in ST.CDS:
        if p == 0:
            return NEG
        return (logs["L_cds"][p - 1]
                + _mix(logs["B_int"], logs["int_coarse"], p, INT_MIX))
    if dst in ST.CDS and src in _INT_GROUP:
        if p == 0:
            return NEG
        return (logs["B_cds"][p]
                + logs["phase"][p, ST.CDS.index(dst) + 1]
                + _mix(logs["L_int"], logs["int_coarse"], p - 1, INT_MIX))
    return em[p, dst]


def _mix(boundary_log, coarse_log, p, alpha):
    """log(exp(boundary) + alpha * exp(coarse)) at position p."""
    b = boundary_log[p]
    c = coarse_log[p] + np.log(alpha)
    m = max(b, c)
    return m + np.log(np.exp(b - m) + np.exp(c - m))


def viterbi_decode(seq, p17, p_phase, p_bnd, len_prior=True,
                   single_gene=False):
    L = len(seq)
    em, donor_ok, acceptor_ok, logs = build_emissions(seq, p17, p_phase,
                                                      p_bnd)
    edges = ST.transitions()
    if single_gene:
        edges = [(s, d) for s, d in edges
                 if s in _SINGLE_GENE_STATES and d in _SINGLE_GENE_STATES]
    group_of = np.array([_group(s) for s in range(ST.N_STATES)])

    start_states = [ST.S0] if single_gene else ST.START_STATES
    end_states = [ST.STOP2] if single_gene else ST.END_STATES

    dp = np.full(ST.N_STATES, NEG, dtype=np.float64)
    for s in start_states:
        dp[s] = 0.0
    dp += em[0]
    btr = np.zeros((L, ST.N_STATES), dtype=np.int8)
    log_bnd = np.log(p_bnd.astype(np.float64) + EPS)
    run_start = np.zeros(ST.N_STATES, dtype=np.int64)

    for p in range(1, L):
        best_prev = np.full(ST.N_STATES, NEG, dtype=np.float64)
        best_src = np.zeros(ST.N_STATES, dtype=np.int8)
        for src, dst in edges:
            splice = (src in ST.CDS and dst in _INT_GROUP) or \
                     (src in _INT_GROUP and dst in ST.CDS)
            if src == dst or (src, dst) in ST.STRUCTURAL_EDGES:
                score = dp[src] + em[p, dst]
            elif splice:
                score = dp[src] + TRANS_LOG \
                    + _edge_emission(src, dst, p, em, logs)
            else:
                score = dp[src] + TRANS_LOG + BND_BONUS * log_bnd[p] \
                    + em[p, dst]
            if len_prior and dp[src] > NEG / 2:
                d = p - run_start[src]
                tables = _length_tables()
                if src in _INT_GROUP and dst not in _INT_GROUP:
                    if tables:
                        score += _len_penalty(d, tables[0])
                    elif d < INTRON_MIN or d > INTRON_MAX:
                        score += LEN_PENALTY
                elif src in _CDS_GROUP and dst in _INT_GROUP:
                    if tables:
                        score += _len_penalty(d, tables[1])
                    elif d < EXON_MIN:
                        score += LEN_PENALTY
            if not _seq_allows(src, dst, p, seq, donor_ok, acceptor_ok):
                continue
            if score > best_prev[dst]:
                best_prev[dst] = score
                best_src[dst] = src
        if len_prior:
            run_start = np.where(
                (best_prev > NEG / 2) & (group_of == group_of[best_src]),
                run_start[best_src], p)
        dp = best_prev
        btr[p] = best_src

    end_state = max(end_states, key=lambda s: dp[s])
    path = np.zeros(L, dtype=np.int8)
    path[-1] = end_state
    for p in range(L - 1, 0, -1):
        path[p - 1] = btr[p][path[p]]
    return path, float(dp[end_state])


def _seq_allows(src, dst, p, seq, donor_ok, acceptor_ok):
    if dst in ST.INT and src in ST.CDS:
        return donor_ok[p] if p + 1 < len(seq) else False
    if dst in ST.CDS and src in ST.INT:
        return seq[p - 2:p] == ST.ACCEPTOR if p >= 2 else False
    return True


def decode_to_genes(path):
    genes = []
    L = len(path)
    p = 0
    while p < L:
        if path[p] in (ST.S0,):
            gene = _parse_gene(path, p)
            if gene:
                genes.append(gene)
                p = gene["end"] + 1
                continue
        p += 1
    return genes


def _parse_gene(path, start):
    L = len(path)
    gene = {"start": start, "cds": [], "introns": [], "u5": None, "u3": None,
            "end": None}
    u5_start = None
    q = start - 1
    while q >= 0 and path[q] == ST.U5:
        u5_start = q
        q -= 1
    if u5_start is not None:
        gene["u5"] = (u5_start, start - 1)

    cur_cds_start = start
    cur_intron_start = None
    p = start
    while p < L:
        s = path[p]
        if s in ST.INT:
            if cur_intron_start is None:
                gene["cds"].append((cur_cds_start, p - 1))
                cur_cds_start = None
                cur_intron_start = p
            p += 1
            continue
        if s in ST.CDS or s in (ST.S0, ST.S1, ST.S2,
                                ST.STOP0, ST.STOP1, ST.STOP2):
            if cur_intron_start is not None:
                gene["introns"].append((cur_intron_start, p - 1))
                cur_intron_start = None
                cur_cds_start = p
            p += 1
            continue
        if cur_intron_start is not None:
            gene["introns"].append((cur_intron_start, p - 1))
            cur_intron_start = None
        if cur_cds_start is not None:
            gene["cds"].append((cur_cds_start, p - 1))
            cur_cds_start = None
        if s == ST.U3:
            u3_start = p
            while p < L and path[p] == ST.U3:
                p += 1
            gene["u3"] = (u3_start, p - 1)
            gene["end"] = p - 1
            return gene
        gene["end"] = p - 1
        return gene
    gene["end"] = L - 1
    return gene if gene["cds"] else None


def rescue_split_genes(seq, genes, p17, p_phase, p_bnd, len_prior=True,
                       max_join=2000):
    """Merge adjacent same-strand genes when a single-transcript path
    (CDS..INT..CDS across the intergenic gap) scores higher than the split.

    Targets the dominant v2 failure mode: a whole intron predicted as
    intergenic, which splits one transcript into two genes. The constrained
    re-decode (single_gene=True: only ATG..CDS..INT..CDS..STOP states) can
    only join when a frame-consistent GT..AG junction exists, so false
    merges require the junction evidence to genuinely outweigh the split.
    """
    genes = sorted(genes, key=lambda g: g["start"])
    merged = []
    i = 0
    while i < len(genes):
        g1 = genes[i]
        if i + 1 < len(genes):
            g2 = genes[i + 1]
            lo = g1["cds"][0][0]
            hi = g2["cds"][-1][1]
            gap = g2["cds"][0][0] - g1["cds"][-1][1] - 1
            if INTRON_MIN <= gap <= max_join and hi - lo < 20000:
                sub = seq[lo:hi + 1]
                p17_sub = p17[lo:hi + 1]
                ph_sub = p_phase[lo:hi + 1]
                bnd_sub = p_bnd[lo:hi + 1]
                _, split_score = viterbi_decode(
                    sub, p17_sub, ph_sub, bnd_sub, len_prior=len_prior)
                join_path, join_score = viterbi_decode(
                    sub, p17_sub, ph_sub, bnd_sub, len_prior=len_prior,
                    single_gene=True)
                if join_score > split_score:
                    new = _parse_gene(join_path, 0)
                    if new and len(new["introns"]) >= 1:
                        new["start"] += lo
                        new["end"] += lo
                        new["cds"] = [(s + lo, e + lo) for s, e in new["cds"]]
                        new["introns"] = [(s + lo, e + lo)
                                          for s, e in new["introns"]]
                        new["u5"] = g1.get("u5")
                        new["u3"] = g2.get("u3")
                        new["strand"] = g1.get("strand", "+")
                        merged.append(new)
                        i += 2
                        continue
        merged.append(g1)
        i += 1
    return merged
