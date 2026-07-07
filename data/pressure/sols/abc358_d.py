import sys
import numpy as np

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); m = int(data[1])
    A = np.array(data[2:2+n], dtype=np.int64)
    B = np.array(data[2+n:2+n+m], dtype=np.int64)
    A.sort()
    B.sort()
    lb = np.searchsorted(A, B, side='left')
    j = np.arange(m)
    c = lb - j
    pos = j + np.maximum.accumulate(c)
    if m == 0 or pos[m-1] < n:
        ans = int(A[pos].sum())
        sys.stdout.write(str(ans) + "\n")
    else:
        sys.stdout.write("-1\n")

main()
