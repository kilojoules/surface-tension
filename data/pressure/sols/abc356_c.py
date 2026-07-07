import sys
import numpy as np

data = sys.stdin.read().split('\n', 1)
first = data[0].split()
N = int(first[0]); M = int(first[1]); K = int(first[2])
rest = data[1] if len(data) > 1 else ''

tokens = np.array(rest.split())
n = tokens.shape[0]

is_letter = (tokens == 'o') | (tokens == 'x')
letter_idx = np.nonzero(is_letter)[0]

# test id for each token: number of letters before this position (letters count as their own test)
seg = np.cumsum(is_letter.astype(np.int64))
test_id = seg - is_letter.astype(np.int64)

# C-value positions: index 0 and one past each letter
cpos = np.concatenate(([0], letter_idx + 1))
cpos = cpos[cpos < n]
is_C = np.zeros(n, dtype=bool)
is_C[cpos] = True

is_key = (~is_letter) & (~is_C)
key_vals = tokens[is_key].astype(np.int64)
key_tests = test_id[is_key]

masks = np.zeros(M, dtype=np.int64)
np.bitwise_or.at(masks, key_tests, (np.int64(1) << (key_vals - 1)))

results = (tokens[is_letter] == 'o')  # length M, ordered

S = 1 << N
subsets = np.arange(S, dtype=np.int64)
pc = np.unpackbits(subsets.astype('>u4').view(np.uint8)).reshape(-1, 32).sum(axis=1)

idx = subsets[None, :] & masks[:, None]
counts = pc[idx]

cond = np.where(results[:, None], counts >= K, counts < K)
valid = np.all(cond, axis=0)
print(int(valid.sum()))
