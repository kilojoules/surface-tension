import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
a = np.array(data[1:1 + n], dtype=np.int64)
b = np.array(data[1 + n:1 + n + (n - 1)], dtype=np.int64)
a.sort()
b.sort()

bad_pre = a[:n - 1] > b          # a_i must be <= b_i for kept prefix
bad_suf = a[1:] > b              # a_{i+1} must be <= b_i for shifted suffix

cum_pre = np.concatenate(([0], np.cumsum(bad_pre)))      # cum_pre[p]=sum bad_pre[:p]
cum_suf = np.concatenate(([0], np.cumsum(bad_suf)))
tot_suf = int(bad_suf.sum())
suf_from_p = tot_suf - cum_suf                            # sum bad_suf[p:]

prefix_ok = cum_pre == 0
suffix_ok = suf_from_p == 0
feasible = prefix_ok & suffix_ok

if bool(feasible.any()):
    print(int(a[feasible].min()))
else:
    print(-1)
