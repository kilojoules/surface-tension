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
    
    # We need to parse M tests. 
    # Each test has C_i, then C_i integers, then R_i.
    # Since we can't use for-loops to iterate through the input stream easily 
    # without knowing indices, we'll use a pointer-like approach with a list.
    
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        # Get the keys (convert to 0-indexed for bitmasking)
        keys = [int(x) - 1 for x in input_data[current + 1 : current + 1 + C_i]]
        # Get the result
        result = input_data[current + 1 + C_i]
        tests.append((keys, result))
        current += 2 + C_i

    # The number of combinations is 2^N. N <= 15, so 2^15 = 32768.
    # We can iterate through all possible combinations using a bitmask.
    # mask i: the j-th bit is 1 if key j is real, 0 if dummy.
    
    # Helper to count set bits (number of real keys in a subset)
    # We use bin(mask & subset_mask).count('1')
    
    # Precompute subset masks for each test to avoid loops inside the comprehension
    test_masks = [
        (sum(1 << k for k in keys), result) 
        for keys, result in tests
    ]
    
    # Iterate through all possible key configurations (2^N)
    # For each configuration, check if it satisfies all M tests.
    # A configuration is valid if:
    # For every test (mask, res):
    #   if res == 'o', then popcount(mask & config) >= K
    #   if res == 'x', then popcount(mask & config) < K
    
    valid_combinations = [
        mask for mask in range(1 << N)
        if all(
            (bin(mask & t_mask).count('1') >= K) if res == 'o' else (bin(mask & t_mask).count('1') < K)
            for t_mask, res in test_masks
        )
    ]
    
    print(len(valid_combinations))

if __name__ == "__main__":
    solve()