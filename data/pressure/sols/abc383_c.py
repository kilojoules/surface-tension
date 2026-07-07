import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

data = sys.stdin.buffer.read().split(b'\n')
first = data[0].split()
H = int(first[0]); W = int(first[1]); D = int(first[2])

rows = data[1:1 + H]
grid = np.frombuffer(b''.join(rows), dtype=np.uint8).reshape(H, W)

passable = grid != ord('#')
ishum = grid == ord('H')

ids = np.arange(H * W).reshape(H, W)

# horizontal edges
hmask = passable[:, :-1] & passable[:, 1:]
ha = ids[:, :-1][hmask]
hb = ids[:, 1:][hmask]

# vertical edges
vmask = passable[:-1, :] & passable[1:, :]
va = ids[:-1, :][vmask]
vb = ids[1:, :][vmask]

S = H * W
hum_ids = ids[ishum].ravel()

rr = np.concatenate([ha, hb, va, vb, np.full(hum_ids.size, S, dtype=ids.dtype)])
cc = np.concatenate([hb, ha, vb, va, hum_ids])
wt = np.concatenate([
    np.ones(ha.size + hb.size + va.size + vb.size, dtype=np.float64),
    np.zeros(hum_ids.size, dtype=np.float64),
])

n = H * W + 1
graph = csr_matrix((wt, (rr, cc)), shape=(n, n))

dist = dijkstra(graph, directed=False, indices=S)

cell_dist = dist[:H * W].reshape(H, W)
ans = int((passable & (cell_dist <= D)).sum())
print(ans)
