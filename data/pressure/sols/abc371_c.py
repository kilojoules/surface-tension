import sys
import numpy as np

data = np.array(sys.stdin.read().split(), dtype=np.int64)
N = int(data[0])
MG = int(data[1])
Gedges = data[2:2 + 2 * MG].reshape(MG, 2)
idx = 2 + 2 * MG
MH = int(data[idx])
Hedges = data[idx + 1:idx + 1 + 2 * MH].reshape(MH, 2)
idx2 = idx + 1 + 2 * MH
rest = data[idx2:]

A = np.zeros((N, N), dtype=np.int64)
iu = np.triu_indices(N, 1)
A[iu] = rest
A = A + A.T

Gadj = np.zeros((N, N), dtype=bool)
Gadj[Gedges[:, 0] - 1, Gedges[:, 1] - 1] = True
Gadj[Gedges[:, 1] - 1, Gedges[:, 0] - 1] = True

Hadj = np.zeros((N, N), dtype=bool)
Hadj[Hedges[:, 0] - 1, Hedges[:, 1] - 1] = True
Hadj[Hedges[:, 1] - 1, Hedges[:, 0] - 1] = True

# generate all permutations of range(N) via cartesian product filtering
grid = np.indices((N,) * N, dtype=np.int8).reshape(N, -1).T
mask = (np.sort(grid, axis=1) == np.arange(N, dtype=np.int8)).all(axis=1)
perms = grid[mask].astype(np.intp)  # (K, N)

pi = perms[:, :, None]
pj = perms[:, None, :]
Hperm = Hadj[pi, pj]          # (K,N,N)
Aperm = A[pi, pj]             # (K,N,N)
diff = Hperm != Gadj[None]
tri = np.triu(np.ones((N, N), dtype=bool), 1)
cost = (Aperm * diff * tri).sum(axis=(1, 2))
print(int(cost.min()))
