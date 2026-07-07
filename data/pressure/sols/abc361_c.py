import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
k = int(data[1])
a = np.sort(np.array(data[2:2+n], dtype=np.int64))
m = n - k
windows = a[m-1:] - a[:n-m+1]
print(int(windows.min()))
