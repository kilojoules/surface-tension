import sys
import numpy as np

data = sys.stdin.buffer.read().split()
n = int(data[0])
H = np.array(data[1:1+n], dtype=np.int64)

c = (H - 1) // 5
sumc = int(c.sum())
rem = (H - 5 * c).astype(np.int64)   # values in 1..5

# table[phase][rem] -> e (extra steps within final partial cycle)
table = {0: {1: 1, 2: 2, 3: 3, 4: 3, 5: 3},
         1: {1: 1, 2: 2, 3: 2, 4: 2, 5: 3},
         2: {1: 1, 2: 1, 3: 1, 4: 2, 5: 3}}

def op(state, r):
    s = int(state)
    phase = s & 3
    esum = s >> 2
    e = table[phase][int(r)]
    return ((esum + e) << 2) | ((phase + e) % 3)

arr = np.empty(n + 1, dtype=object)
arr[0] = 0                 # initial packed state: phase=0, esum=0
arr[1:] = rem              # remaining entries are rem values (1..5)

final = np.frompyfunc(op, 2, 1).accumulate(arr)[-1]
esum = int(final) >> 2

T = 3 * sumc + esum
sys.stdout.write(str(T) + "\n")
