import sys
import numpy as np

N = int(sys.stdin.readline())
size = 3 ** N
idx = np.arange(size)
powers = 3 ** np.arange(N)
rmid = ((idx[:, None] // powers) % 3 == 1)
cmid = ((idx[:, None] // powers) % 3 == 1)
white = (rmid.astype(np.int64) @ cmid.astype(np.int64).T) > 0
grid = np.where(white, 46, 35).astype(np.uint8)
out = np.column_stack([grid, np.full((size, 1), 10, dtype=np.uint8)])
sys.stdout.write(out.tobytes().decode())
