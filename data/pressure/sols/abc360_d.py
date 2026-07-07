import sys
import numpy as np

def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0]); T = int(data[1])
    s = data[2].decode()
    x = np.array(data[3:3+n], dtype=np.int64)
    dirs = np.frombuffer(s.encode(), dtype=np.uint8) - ord('0')
    ones = np.sort(x[dirs == 1])
    zeros = x[dirs == 0]
    lo = np.searchsorted(ones, zeros - 2*T, side='left')
    hi = np.searchsorted(ones, zeros, side='left')
    print(int((hi - lo).sum()))

main()
