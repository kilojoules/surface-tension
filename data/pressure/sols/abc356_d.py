import sys
import numpy as np

def main():
    parts = sys.stdin.read().split()
    N = int(parts[0])
    M = int(parts[1])
    mod = 998244353
    b = np.arange(60, dtype=object)
    p = 2 ** b
    period = 2 * p
    full = (N + 1) // period
    count = full * p
    rem = (N + 1) % period
    extra = np.maximum(0, rem - p)
    count = count + extra
    mask = (M >> b) & 1
    total = int((count * mask).sum()) % mod
    print(total)

main()
