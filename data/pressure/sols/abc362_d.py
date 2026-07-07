import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

d = np.array(sys.stdin.buffer.read().split(), dtype=np.int64)
N = int(d[0]); M = int(d[1])
A = d[2:2 + N]
rest = d[2 + N:2 + N + 3 * M].reshape(M, 3)
U = rest[:, 0] - 1
V = rest[:, 1] - 1
B = rest[:, 2]

rows = np.concatenate([U, V])
cols = np.concatenate([V, U])
wdata = np.concatenate([B + A[V], B + A[U]]).astype(np.float64)

g = csr_matrix((wdata, (rows, cols)), shape=(N, N))
dist = dijkstra(g, indices=0)
ans = (dist[1:] + A[0]).astype(np.int64)
sys.stdout.write(' '.join(ans.astype(str)) + '\n')
