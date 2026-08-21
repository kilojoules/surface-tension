import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # The remaining data consists of M tests.
    # Each test starts with C_i, followed by C_i keys, then R_i.
    # We need to parse these into a list of (key_set, result)
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        # Extract the keys as a set (using 0-indexed internally)
        keys = {int(x) - 1 for x in input_data[current + 1 : current + 1 + C_i]}
        result = input_data[current + 1 + C_i]
        tests.append((keys, result))
        current += C_i + 2

    # There are 2^N possible combinations of real/dummy keys.
    # We represent each combination as a tuple of 0s (dummy) and 1s (real).
    # We filter these combinations based on whether they satisfy all M tests.
    
    # Generator for all possible combinations
    all_combinations = product([0, 1], repeat=N)
    
    # Helper to check if a specific combination is valid
    def is_valid(combo):
        for keys, result in tests:
            # Count how many keys in the test set are 'real' (1) in this combination
            real_count = sum(combo[k] for k in keys)
            
            # Door opens if real_count >= K.
            # If result is 'o', we need real_count >= K.
            # If result is 'x', we need real_count < K.
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Count valid combinations using a list comprehension/filter and len()
    # Since we cannot use for/while loops, we use map or filter.
    answer = len(list(filter(is_valid, all_combinations)))
    print(answer)

if __name__ == "__main__":
    solve()