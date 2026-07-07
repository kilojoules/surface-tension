import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

data = sys.stdin.buffer.read().split()
n = int(data[0]); m = int(data[1])
rest = np.array(data[2:2+2*m], dtype=np.int64)
a = rest[0::2] - 1
b = rest[1::2] - 1

g = csr_matrix((np.ones(m), (a, b)), shape=(n, n))
dist = dijkstra(g, directed=True, indices=0)

mask = (b == 0)
cand = dist[a[mask]] + 1.0
if cand.size == 0 or not np.isfinite(cand).any():
    print(-1)
else:
    print(int(np.min(cand)))
