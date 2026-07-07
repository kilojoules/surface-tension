import sys
import numpy as np

data = sys.stdin.read().split('\n')
H, W, D = np.array(data[0].split(), dtype=np.int64)
grid = ''.join(data[1:1 + int(H)])
arr = np.frombuffer(grid.encode(), dtype=np.uint8)
floor = np.nonzero(arr == ord('.'))[0]
rows = floor // int(W)
cols = floor % int(W)
dist = np.abs(rows[:, None] - rows[None, :]) + np.abs(cols[:, None] - cols[None, :])
cover = dist <= D
union = cover[:, None, :] | cover[None, :, :]
counts = union.sum(axis=2)
print(int(counts.max()))
