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
    
    # The input format for tests is variable length: C_i, then C_i keys, then R_i
    # We need to parse the tests carefully.
    tests = []
    current_pos = 3
    for _ in range(M):
        C_i = int(input_data[current_pos])
        # Extract the keys (adjusting to 0-indexed for easier bit manipulation/indexing)
        keys = [int(x) - 1 for x in input_data[current_pos + 1 : current_pos + 1 + C_i]]
        # Extract the result
        result = input_data[current_pos + 1 + C_i]
        tests.append((keys, result))
        # Move pointer to the start of the next test
        current_pos += (C_i + 2)

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # - If result is 'o', the number of real keys in the set is >= K
    # - If result is 'x', the number of real keys in the set is < K
    # We use a generator expression inside sum() to count valid combinations.
    
    valid_count = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k] for k in test_keys) < K)
            for (test_keys, res) in tests
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()