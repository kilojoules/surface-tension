import sys
import numpy as np

def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0]); Q = int(data[1])
    S = data[2]
    Scodes = np.frombuffer(S, dtype=np.uint8).astype(np.int64)  # length N, position p -> Scodes[p-1]

    A = ord('A'); B = ord('B'); C = ord('C')

    # initial count of ABC windows
    if N >= 3:
        init = int(((Scodes[:-2] == A) & (Scodes[1:-1] == B) & (Scodes[2:] == C)).sum())
    else:
        init = 0

    if Q == 0:
        return

    # parse queries
    Xtok = data[3::2][:Q]
    Ctok = data[4::2][:Q]
    X = np.array(Xtok, dtype='S').astype(np.int64)  # 1-indexed positions
    Cbytes = b''.join(Ctok)
    Ccodes = np.frombuffer(Cbytes, dtype=np.uint8).astype(np.int64)

    times = np.arange(Q, dtype=np.int64)
    base = np.int64(Q + 1)

    # updates keyed by position*base + time
    ukey = X * base + times
    order = np.argsort(ukey, kind='stable')
    ukey_s = ukey[order]
    uval_s = Ccodes[order]

    def lookup(I, P):
        # value at position P (1-indexed, clamped [1,N]) just before query index I
        lk = P * base + I
        idx = np.searchsorted(ukey_s, lk, side='left')
        cand = idx - 1
        candc = np.clip(cand, 0, len(ukey_s) - 1)
        match = (cand >= 0) & (ukey_s[candc] // base == P)
        return np.where(match, uval_s[candc], Scodes[P - 1])

    I = times  # query index for each query
    def clampP(p):
        return np.clip(p, 1, N)

    Pa = X - 2; Pb = X - 1; Pc = X; Pd = X + 1; Pe = X + 2
    va = lookup(I, clampP(Pa))
    vb = lookup(I, clampP(Pb))
    vc_old = lookup(I, clampP(Pc))
    vd = lookup(I, clampP(Pd))
    ve = lookup(I, clampP(Pe))
    vc_new = Ccodes

    # window validity (start s valid if 1<=s<=N-2)
    w1_ok = (Pa >= 1) & (Pa <= N - 2)
    w2_ok = (Pb >= 1) & (Pb <= N - 2)
    w3_ok = (Pc >= 1) & (Pc <= N - 2)

    def isabc(x, y, z):
        return ((x == A) & (y == B) & (z == C)).astype(np.int64)

    before = (w1_ok * isabc(va, vb, vc_old)
              + w2_ok * isabc(vb, vc_old, vd)
              + w3_ok * isabc(vc_old, vd, ve))
    after = (w1_ok * isabc(va, vb, vc_new)
             + w2_ok * isabc(vb, vc_new, vd)
             + w3_ok * isabc(vc_new, vd, ve))

    delta = after - before
    ans = init + np.cumsum(delta)

    sys.stdout.write('\n'.join(ans.astype('U')) + '\n')

main()
