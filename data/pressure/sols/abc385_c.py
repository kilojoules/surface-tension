import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
H = np.array(data[1:1 + n], dtype=np.int32)

if n <= 1:
    print(1)
    sys.exit(0)

d = np.arange(1, n)                       # candidate steps
k = np.arange(n)
D = np.mod(k[None, :], d[:, None])        # residue of each position for each step
key = D.astype(np.int64) * n + k[None, :]
P = np.argsort(key, axis=1)               # positions in chain order, per step

ordpos = P.reshape(-1)
M = ordpos.size
idx = np.arange(M)
drow = idx // n + 1                        # step d for each flattened slot
res = ordpos % drow                        # residue (chain id within its step)
Hord = H[ordpos]

rowstart = (idx % n) == 0

prev_res = np.empty(M, dtype=res.dtype)
prev_res[1:] = res[:-1]
prev_res[0] = -1
prev_H = np.empty(M, dtype=Hord.dtype)
prev_H[1:] = Hord[:-1]
prev_H[0] = -1

boundary = rowstart | (res != prev_res) | (Hord != prev_H)
ref = np.where(boundary, idx, 0)
segstart = np.maximum.accumulate(ref)
runlen = idx - segstart + 1

print(int(runlen.max()))
