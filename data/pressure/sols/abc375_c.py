import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
buf = b''.join(data[1:1 + n])
arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, n)

# ring index (1-based) for each cell = distance-from-border + 1
rows = np.arange(n)[:, None]
cols = np.arange(n)[None, :]
d = np.minimum(np.minimum(rows, cols), np.minimum(n - 1 - rows, n - 1 - cols))
q = (d + 1) % 4  # number of clockwise quarter-turns for this cell's ring

# clockwise-by-q rotations of the whole grid (rotation preserves each ring)
R0 = arr
R1 = np.rot90(arr, -1)
R2 = np.rot90(arr, -2)
R3 = np.rot90(arr, -3)

out = np.choose(q, [R0, R1, R2, R3]).astype(np.uint8)

nl = np.full((n, 1), 10, dtype=np.uint8)
sys.stdout.buffer.write(np.hstack([out, nl]).tobytes())
