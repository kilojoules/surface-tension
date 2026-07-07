import sys, re
import numpy as np


def main():
    text = sys.stdin.buffer.read().decode()
    nl = text.index('\n')
    rest = text[nl + 1:]
    rest = re.sub(r'(?m)^3\s*$', '3 0', rest)
    arr = np.array(rest.split(), dtype=np.int64).reshape(-1, 2)
    typ = arr[:, 0]
    val = arr[:, 1]
    Q = typ.shape[0]
    sgn = np.where(typ == 1, 1, np.where(typ == 2, -1, 0)).astype(np.int64)
    op_idx = np.flatnonzero(typ != 3)
    d_full = np.zeros(Q, dtype=np.int64)
    if op_idx.shape[0] > 0:
        v = val[op_idx]
        s = sgn[op_idx]
        perm = np.argsort(v, kind='stable')
        vs = v[perm]
        ss = s[perm]
        fc = np.cumsum(ss)
        is_start = np.empty(vs.shape[0], dtype=bool)
        is_start[0] = True
        is_start[1:] = vs[1:] != vs[:-1]
        group_id = np.cumsum(is_start) - 1
        starts = np.flatnonzero(is_start)
        carry_vals = np.where(starts == 0, 0, fc[starts - 1])
        carry = carry_vals[group_id]
        incl = fc - carry
        before = incl - ss
        d_sorted = np.where(ss == 1, (before == 0).astype(np.int64),
                            np.where(incl == 0, -1, 0))
        d_ops = np.empty(op_idx.shape[0], dtype=np.int64)
        d_ops[perm] = d_sorted
        d_full[op_idx] = d_ops
    pref = np.cumsum(d_full)
    ans = pref[typ == 3]
    sys.stdout.write('\n'.join(ans.astype(str)) + '\n')


main()
