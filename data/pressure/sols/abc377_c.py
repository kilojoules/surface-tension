import sys
import numpy as np

data = sys.stdin.buffer.read().split()
vals = np.array(data, dtype=np.int64)
N = int(vals[0])
M = int(vals[1])
rest = vals[2:2 + 2 * M].reshape(M, 2)
a = rest[:, 0]
b = rest[:, 1]

dx = np.array([2, 1, -1, -2, -2, -1, 1, 2], dtype=np.int64)
dy = np.array([1, 2, 2, 1, -1, -2, -2, -1], dtype=np.int64)

ta = a[:, None] + dx[None, :]
tb = b[:, None] + dy[None, :]
mask = (ta >= 1) & (ta <= N) & (tb >= 1) & (tb <= N)

codes = ((ta - 1) * N + (tb - 1))[mask]
attacked = np.unique(codes)

occ = (a - 1) * N + (b - 1)
forbidden_empty = int((~np.isin(attacked, occ)).sum())

ans = N * N - M - forbidden_empty
sys.stdout.write(str(ans) + "\n")
