import sys
import numpy as np

data = sys.stdin.buffer.read().split()
N = int(data[0]); M = int(data[1])
A = np.array(data[2:2 + N], dtype=np.int64)

P = np.empty(N, dtype=np.int64)
P[0] = 0
P[1:] = np.cumsum(A[:-1])
Total = int(A.sum())
c = (P % M).astype(np.int64)

cnt = np.bincount(c, minlength=M)
S1 = int(np.sum(cnt * (cnt - 1) // 2))
T = Total % M

if T == 0:
    ans = int(np.sum(cnt * (cnt - 1)))
else:
    leftkey = (c + T) % M
    rightkey = c
    key_all = np.concatenate([leftkey, rightkey])
    pos_all = np.concatenate([np.arange(N), np.arange(N)])
    isleft = np.concatenate([np.ones(N, dtype=np.int64), np.zeros(N, dtype=np.int64)])
    order = np.lexsort((pos_all, key_all))
    k_s = key_all[order]
    isleft_s = isleft[order]
    twoN = 2 * N
    starts = np.ones(twoN, dtype=bool)
    starts[1:] = k_s[1:] != k_s[:-1]
    group_id = np.cumsum(starts) - 1
    pref = np.concatenate([[0], np.cumsum(isleft_s)])
    first_index_of_group = np.nonzero(starts)[0]
    group_base = pref[first_index_of_group]
    within_before = pref[:twoN] - group_base[group_id]
    B = int(np.sum(within_before[isleft_s == 0]))
    ans = S1 + B

print(ans)
