import sys
import numpy as np

NEG = -1e15
base = np.array([[0, 1, NEG], [NEG, 0, 1], [1, NEG, 0]], dtype=np.float64)


def combine(arr):
    m = arr.shape[0]
    arr = arr.reshape(m // 2, 2, 3, 3)
    Q = arr[:, 0]
    P = arr[:, 1]
    return (P[:, :, :, None] + Q[:, None, :, :]).max(axis=2)


data = sys.stdin.buffer.read().split()
S = data[1]
s = np.frombuffer(S.translate(bytes.maketrans(b'RPS', bytes([0, 1, 2]))),
                  dtype=np.uint8).astype(np.int64)
N = s.shape[0]
val = base[s]
M = np.broadcast_to(val[:, :, None], (N, 3, 3)).copy()
di = np.arange(3)
M[:, di, di] = NEG
M[0] = np.broadcast_to(val[0][:, None], (3, 3))

P2 = 262144
ident = np.full((3, 3), NEG)
ident[di, di] = 0.0
arr = np.broadcast_to(ident, (P2, 3, 3)).copy()
arr[:N] = M

arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)
arr = combine(arr)

print(int(round(arr[0].max())))
