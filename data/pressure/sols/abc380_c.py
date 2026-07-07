import sys
import numpy as np

data = sys.stdin.buffer.read().split()
K = int(data[1])
S = data[2]
arr = np.frombuffer(S, dtype=np.uint8) - 48
pad = np.concatenate(([0], arr, [0])).astype(np.int8)
diff = np.diff(pad)
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0] - 1
lK = starts[K - 1]
rK = ends[K - 1]
rKm1 = ends[K - 2]
block_len = rK - lK + 1
res = arr.copy()
a = rKm1 + 1
b = rK + 1
res[a:b] = 0
res[a:a + block_len] = 1
sys.stdout.write((res + 48).astype(np.uint8).tobytes().decode())
