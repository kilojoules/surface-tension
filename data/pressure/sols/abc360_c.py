import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
A = np.array(data[1:1+n], dtype=np.int64)
W = np.array(data[1+n:1+2*n], dtype=np.int64)

group_max = np.zeros(n + 1, dtype=np.int64)
np.maximum.at(group_max, A, W)
print(int(W.sum() - group_max.sum()))
