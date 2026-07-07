import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
a = np.array(data[1:1 + n], dtype=np.int64)

order = np.argsort(a, kind='stable')
sorted_a = a[order]
same = sorted_a[1:] == sorted_a[:-1]

B = np.full(n, -1, dtype=np.int64)
targets = order[1:]
cand = order[:-1] + 1
B[targets[same]] = cand[same]

sys.stdout.write(' '.join(B.astype(str)))
