IG, U5 = 0, 1
S0, S1, S2 = 2, 3, 4
CDS = [5, 6, 7]
INT = [8, 9, 10]
STOP0, STOP1, STOP2 = 11, 12, 13
U3 = 14
N_STATES = 15

STATE_NAMES = {IG: "IG", U5: "U5", S0: "S0", S1: "S1", S2: "S2",
               CDS[0]: "CDS0", CDS[1]: "CDS1", CDS[2]: "CDS2",
               INT[0]: "INT0", INT[1]: "INT1", INT[2]: "INT2",
               STOP0: "STOP0", STOP1: "STOP1", STOP2: "STOP2", U3: "U3"}

CLS_IG, CLS_UTR, CLS_CDS, CLS_INTRON = 0, 1, 2, 3

STOP_CODONS = {"TAA", "TAG", "TGA"}
DONORS = {"GT", "GC", "AT"}
ACCEPTOR = "AG"


def transitions():
    edges = []
    add = edges.append
    add((IG, IG))
    add((IG, U5))
    add((IG, S0))
    add((U5, U5))
    add((U5, S0))
    add((S0, S1))
    add((S1, S2))
    add((S2, CDS[0]))
    add((CDS[0], CDS[1]))
    add((CDS[1], CDS[2]))
    add((CDS[2], CDS[0]))
    add((CDS[2], STOP0))
    for i in range(3):
        add((CDS[i], INT[i]))
        add((INT[i], INT[i]))
        add((INT[i], CDS[(i + 1) % 3]))
    add((STOP0, STOP1))
    add((STOP1, STOP2))
    add((STOP2, U3))
    add((STOP2, IG))
    add((U3, U3))
    add((U3, IG))
    return edges


START_STATES = [IG, U5, S0]
END_STATES = [IG, U3, STOP2]

STRUCTURAL_EDGES = {
    (S0, S1), (S1, S2), (S2, CDS[0]),
    (CDS[0], CDS[1]), (CDS[1], CDS[2]), (CDS[2], CDS[0]),
    (STOP0, STOP1), (STOP1, STOP2),
}
