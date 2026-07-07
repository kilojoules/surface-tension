import sys, io
import numpy as np

data = sys.stdin.buffer.read().split()
N = int(data[0]); K = int(data[1])
R = np.array(data[2:2 + N], dtype=np.int64)

total = int(R.prod())
rev = R[::-1]
cp = np.cumprod(rev)
w_rev = np.concatenate(([1], cp[:-1]))
weights = w_rev[::-1].astype(np.int64)

t = np.arange(total, dtype=np.int64)
grid = (t[:, None] // weights[None, :]) % R[None, :] + 1
sums = grid.sum(axis=1)
sel = grid[sums % K == 0]

buf = io.StringIO()
np.savetxt(buf, sel, fmt='%d')
sys.stdout.write(buf.getvalue())
