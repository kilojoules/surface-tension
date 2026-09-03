import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream
    it = iter(input_data)
    
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    
    tests = []
    for _ in range(M):
        C_i = int(next(it))
        # Get the next C_i elements as keys
        keys = [int(next(it)) for _ in range(C_i)]
        # Get the result character
        result = next(it)
        tests.append((keys, result))
    
    # The number of possible combinations is 2^N.
    # Since N <= 15, 2^15 = 32768, which is small enough to brute force.
    # We represent each combination as a tuple of 0s (dummy) and 1s (real).
    
    # We use a generator to avoid loading all combinations into memory
    # and a sum() to count how many satisfy all conditions.
    
    # For a combination 'comb', comb[i-1] is 1 if key i is real, 0 otherwise.
    # The condition for test (keys, result):
    # If result == 'o', sum(comb[k-1] for k in keys) >= K
    # If result == 'x', sum(comb[k-1] for k in keys) < K
    
    # To speed up, we can pre-process the tests into a format that avoids loops.
    # However, with N=15, nested comprehensions are acceptable.
    
    valid_combinations_count = sum(
        1 for comb in product([0, 1], repeat=N)
        if all(
            (sum(comb[k-1] for k in keys) >= K) if res == 'o' else (sum(comb[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(valid_combinations_count)

if __name__ == "__main__":
    solve()