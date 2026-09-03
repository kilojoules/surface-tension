import sys
from itertools import product

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse M tests. 
    # Since the number of keys C_i varies, we can't use a simple range.
    # We'll use a pointer-like approach with an iterator.
    it = iter(input_data[3:])
    
    tests = []
    for _ in range(M):
        # Get C_i
        C_i = int(next(it))
        # Get the next C_i elements as keys
        keys = [int(next(it)) for _ in range(C_i)]
        # Get the result R_i
        result = next(it)
        tests.append((keys, result))
    
    # The number of combinations is 2^N. 
    # Since N <= 15, 2^15 = 32,768, which is small enough to brute force.
    # We represent each combination as a tuple of 0s (dummy) and 1s (real).
    
    # Use a generator to avoid loading all combinations into memory
    # product([0, 1], repeat=N) generates all binary strings of length N
    combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If R_i == 'o', sum of real keys in the set >= K
    # If R_i == 'x', sum of real keys in the set < K
    
    # We use a list comprehension inside sum() to count valid combinations.
    # Note: Key indices in input are 1-based, so we use key-1 for 0-based indexing.
    
    ans = sum(
        1 for combo in combinations
        if all(
            (sum(combo[k-1] for k in keys) >= K) if res == 'o' else (sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()