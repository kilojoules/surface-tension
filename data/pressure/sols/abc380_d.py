import sys
import numpy as np

data = sys.stdin.buffer.read().split()
S = data[0]
S_arr = np.frombuffer(S, dtype=np.uint8)
L = len(S)
Q = int(data[1])
K = np.array(data[2:2 + Q], dtype=np.int64)

p = K - 1
offset = p % L
block = (p // L).astype(np.uint64)

b = block ^ (block >> np.uint64(32))
b = b ^ (b >> np.uint64(16))
b = b ^ (b >> np.uint64(8))
b = b ^ (b >> np.uint64(4))
b = b ^ (b >> np.uint64(2))
b = b ^ (b >> np.uint64(1))
parity = (b & np.uint64(1)).astype(np.uint8)

base = S_arr[offset]
flipped = base ^ np.uint8(32)
res = np.where(parity == 1, flipped, base).astype(np.uint8)

out = np.empty(res.size * 2, dtype=np.uint8)
out[0::2] = res
out[1::2] = 32
sys.stdout.buffer.write(out[:-1].tobytes())
sys.stdout.buffer.write(b"\n")
