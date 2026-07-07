import sys
import numpy as np

data = np.array(sys.stdin.buffer.read().split(), dtype=np.int64)
N = int(data[0])
vals = data[1:1 + N * N * N].reshape(N, N, N)
idx = 1 + N * N * N
Q = int(data[idx])
qs = data[idx + 1:idx + 1 + 6 * Q].reshape(Q, 6)

C = vals.cumsum(0).cumsum(1).cumsum(2)
P = np.zeros((N + 1, N + 1, N + 1), dtype=np.int64)
P[1:, 1:, 1:] = C

lx = qs[:, 0]; rx = qs[:, 1]; ly = qs[:, 2]; ry = qs[:, 3]; lz = qs[:, 4]; rz = qs[:, 5]
lx0 = lx - 1; ly0 = ly - 1; lz0 = lz - 1

res = (P[rx, ry, rz]
       - P[lx0, ry, rz] - P[rx, ly0, rz] - P[rx, ry, lz0]
       + P[lx0, ly0, rz] + P[lx0, ry, lz0] + P[rx, ly0, lz0]
       - P[lx0, ly0, lz0])

np.savetxt(sys.stdout, res, fmt='%d')
