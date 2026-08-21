import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of keys, M: number of tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests
    # Each test is a tuple: (set_of_keys, result)
    # We use a helper to slice the input list based on the C_i values
    tests = []
    current_pos = 3
    for _ in range(M):
        C_i = int(input_data[current_pos])
        keys = set(map(int, input_data[current_pos + 1 : current_pos + 1 + C_i]))
        result = input_data[current_pos + 1 + C_i]
        tests.append((keys, result))
        current_pos += C_i + 2

    # Generate all 2^N possible combinations of keys (0 = dummy, 1 = real)
    # We use a list comprehension to filter combinations that satisfy all M tests
    # For each combination 'comb', we check:
    # If result is 'o', the number of real keys in the test set must be >= K
    # If result is 'x', the number of real keys in the test set must be < K
    
    # We map the combination tuple to a dictionary for easy lookup: {key_id: status}
    # However, since N is small, we can use the index of the tuple directly.
    # Note: keys are 1-indexed, so we use comb[key-1]
    
    all_combinations = product([0, 1], repeat=N)
    
    valid_combinations_count = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if result == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )
    
    print(valid_combinations_count)

if __name__ == "__main__":
    solve()