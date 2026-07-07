import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
S = np.frombuffer(data[1], dtype=np.uint8)

ar = np.arange(n)
one = (S == ord('1'))
two = (S == ord('2'))
sl = (S == ord('/'))

# run of 1's ending at each index
r1 = ar - np.maximum.accumulate(np.where(~one, ar, -1))
# run of 2's starting at each index (compute on reversed)
tr = two[::-1]
rr = ar - np.maximum.accumulate(np.where(~tr, ar, -1))
r2 = rr[::-1]

# ones ending just before index i
sb = np.zeros(n, dtype=np.int64)
sb[1:] = r1[:-1]
# twos starting just after index i
sa = np.zeros(n, dtype=np.int64)
sa[:-1] = r2[1:]

cand = 2 * np.minimum(sb, sa) + 1
print(int(cand[sl].max()))
