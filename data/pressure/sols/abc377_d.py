import sys
import numpy as np

def main():
    data = sys.stdin.buffer.read().split()
    N = int(data[0]); M = int(data[1])
    vals = np.array(data[2:2 + 2 * N], dtype=np.int64)
    L = vals[0::2]
    R = vals[1::2]
    arr = np.zeros(M + 1, dtype=np.int64)
    np.maximum.at(arr, R, L)
    fcum = np.maximum.accumulate(arr)
    r = np.arange(0, M + 1, dtype=np.int64)
    ans = int((r[1:] - fcum[1:]).sum())
    sys.stdout.write(str(ans) + "\n")

main()
