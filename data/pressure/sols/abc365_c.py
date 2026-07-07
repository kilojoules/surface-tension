import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
m = int(data[1])
a = np.sort(np.array(data[2:2 + n], dtype=np.int64))
total = int(a.sum())
if total <= m:
    print("infinite")
else:
    prefix = np.concatenate(([0], np.cumsum(a)[:-1])).astype(np.int64)
    idx = np.arange(n, dtype=np.int64)
    cost_at = prefix + a * (n - idx)
    j = int(np.argmax(cost_at > m))
    ans = (m - int(prefix[j])) // (n - j)
    print(ans)
