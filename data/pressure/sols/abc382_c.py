import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
m = int(data[1])
A = np.array(data[2:2 + n], dtype=np.int64)
B = np.array(data[2 + n:2 + n + m], dtype=np.int64)

pm = np.minimum.accumulate(A)
neg = -pm
idx = np.searchsorted(neg, -B, side='left')
ans = np.where(idx == n, -1, idx + 1)
sys.stdout.write("\n".join(ans.astype(str).tolist()) + "\n")
