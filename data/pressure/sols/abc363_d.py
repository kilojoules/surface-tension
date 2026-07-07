import sys
import numpy as np

n = int(sys.stdin.buffer.read().split()[0])

L = np.arange(1, 42)
h = (L + 1) // 2
cnt = 9 * (10 ** (h.astype(object) - 1))
cnt[0] = 10  # length-1 palindromes: 0..9
cum = np.cumsum(cnt)

idx_len = int(np.searchsorted(cum, n, side='left'))
cum_before = int(cum[idx_len - 1]) if idx_len > 0 else 0
offset = n - cum_before
Lval = int(L[idx_len])
hh = (Lval + 1) // 2

half = (offset - 1) if Lval == 1 else (10 ** (hh - 1) + (offset - 1))
s = str(half)
if Lval % 2 == 0:
    pal = s + s[::-1]
else:
    pal = s + s[:-1][::-1]

sys.stdout.write(pal + "\n")
