import sys
import numpy as np

data = sys.stdin.buffer.read().split()
idx = 0
n = int(data[0])
X = np.array(data[1:1+n], dtype=np.int64)
P = np.array(data[1+n:1+2*n], dtype=np.int64)
q = int(data[1+2*n])
rest = np.array(data[2+2*n:2+2*n+2*q], dtype=np.int64)
L = rest[0::2]
R = rest[1::2]

pre = np.zeros(n+1, dtype=np.int64)
pre[1:] = np.cumsum(P)

li = np.searchsorted(X, L, side='left')
ri = np.searchsorted(X, R, side='right')
ans = pre[ri] - pre[li]

sys.stdout.write('\n'.join(ans.astype(str)) + '\n')
