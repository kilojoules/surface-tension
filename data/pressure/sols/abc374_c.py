import sys
import numpy as np

data = sys.stdin.read().split()
n = int(data[0])
k = np.array(data[1:1 + n], dtype=np.int64)
total = int(k.sum())

masks = np.arange(1 << n, dtype=np.int64)
bits = ((masks[:, None] >> np.arange(n, dtype=np.int64)) & 1)
sums = bits @ k
worst = np.maximum(sums, total - sums)
print(int(worst.min()))
