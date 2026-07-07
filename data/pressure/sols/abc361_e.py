import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    if n < 2:
        print(0)
        return
    rest = np.array(data[1:], dtype=np.int64).reshape(n - 1, 3)
    a = rest[:, 0] - 1
    b = rest[:, 1] - 1
    c = rest[:, 2]
    total = int(c.sum())
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    vals = np.concatenate([c, c]).astype(np.float64)
    g = csr_matrix((vals, (rows, cols)), shape=(n, n))
    d0 = shortest_path(g, method='D', indices=0)
    u = int(np.argmax(d0))
    du = shortest_path(g, method='D', indices=u)
    diameter = int(round(du.max()))
    print(2 * total - diameter)

main()
