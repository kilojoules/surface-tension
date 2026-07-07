import sys
import numpy as np

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); X = int(data[1]); Y = int(data[2])
    A = np.array(data[3:3+n], dtype=np.int64)
    B = np.array(data[3+n:3+2*n], dtype=np.int64)
    A = np.sort(A)[::-1]
    B = np.sort(B)[::-1]
    csA = np.cumsum(A)
    csB = np.cumsum(B)
    idxA = int(np.searchsorted(csA, X, side='right'))
    idxB = int(np.searchsorted(csB, Y, side='right'))
    kA = min(idxA + 1, n)
    kB = min(idxB + 1, n)
    print(min(kA, kB))

main()
