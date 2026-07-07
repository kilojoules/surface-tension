import sys
import numpy as np

data = sys.stdin.read().split()
N = int(data[0]); M = int(data[1])
s = ''.join(data[2:2+N])
arr = np.frombuffer(s.encode(), dtype=np.uint8).reshape(N, M)
grid = (arr == ord('o')).astype(np.int64)  # (N, M)

subsets = np.arange(1 << N)
included = ((subsets[:, None] >> np.arange(N)) & 1)  # (S, N)
cover_counts = included @ grid  # (S, M): number of chosen stands selling flavor j
full = (cover_counts > 0).all(axis=1)  # subsets covering all flavors
popcount = included.sum(axis=1)
best = np.where(full, popcount, N + 1).min()
print(int(best))
