import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
a = np.array(data[1:1+n], dtype=np.int64)

if n == 1:
    print(1)
else:
    d = np.diff(a)
    m = d.shape[0]
    idx = np.flatnonzero(d[1:] != d[:-1]) + 1
    boundaries = np.concatenate(([0], idx, [m])).astype(np.int64)
    runs = np.diff(boundaries)
    ap = (runs * (runs + 1) // 2).sum()
    print(int(n) + int(ap))
