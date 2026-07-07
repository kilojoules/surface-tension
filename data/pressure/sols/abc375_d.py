import sys
import numpy as np

S = sys.stdin.readline().strip()
n = len(S)
if n < 3:
    print(0)
else:
    codes = np.frombuffer(S.encode(), dtype=np.uint8).astype(np.int64) - ord('A')
    idx = np.arange(n, dtype=np.int64)
    letters = np.arange(26, dtype=np.int64)
    onehot = (codes[None, :] == letters[:, None])
    cumcnt = np.cumsum(onehot.astype(np.int64), axis=1)
    cumsumidx = np.cumsum(onehot * idx[None, :], axis=1)
    cnt_incl = cumcnt[codes, idx]
    sum_incl = cumsumidx[codes, idx]
    cnt_before = cnt_incl - 1
    sum_before = sum_incl - idx
    contrib = cnt_before * (idx - 1) - sum_before
    print(int(contrib.sum()))
