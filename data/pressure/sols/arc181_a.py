import sys
import numpy as np

def main():
    lines = sys.stdin.buffer.read().split(b'\n')
    T = int(lines[0])
    nlines = lines[1:2*T+1:2]
    plines = lines[2:2*T+1:2]
    lengths = np.array(nlines, dtype=np.int64)          # N_i per case
    total = int(lengths.sum())
    A = np.array(b' '.join(plines).split(), dtype=np.int64)  # concatenated perms

    starts = np.zeros(T, dtype=np.int64)
    starts[1:] = np.cumsum(lengths)[:-1]
    seg = np.repeat(np.arange(T, dtype=np.int64), lengths)
    localpos = np.arange(total, dtype=np.int64) - starts[seg] + 1  # 1-based within case

    BIG = np.int64(1 << 20)
    cummax = np.maximum.accumulate(A + seg * BIG) - seg * BIG      # inclusive prefix max within case

    good = (A == localpos) & (cummax == localpos)
    good_any = np.bincount(seg, weights=good.astype(np.float64), minlength=T) > 0.5

    fixed = (A == localpos)
    fixed_cnt = np.bincount(seg, weights=fixed.astype(np.float64), minlength=T)
    identity = fixed_cnt == lengths.astype(np.float64)

    firstN = A[starts] == lengths
    lastis1 = A[starts + lengths - 1] == 1
    case3 = firstN & lastis1

    ans = np.full(T, 2, dtype=np.int64)
    ans = np.where(case3, 3, ans)
    ans = np.where(good_any, 1, ans)
    ans = np.where(identity, 0, ans)

    sys.stdout.write('\n'.join(ans.astype(str).tolist()))
    sys.stdout.write('\n')

main()
