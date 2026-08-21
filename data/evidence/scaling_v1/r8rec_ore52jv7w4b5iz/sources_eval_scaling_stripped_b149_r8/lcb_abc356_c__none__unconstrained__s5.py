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

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a helper function to extract tests from the flat list
    def extract_tests(data, m_count):
        tests = []
        current_idx = 3
        for _ in range(m_count):
            c_i = int(data[current_idx])
            keys = set(map(int, data[current_idx + 1 : current_idx + 1 + c_i]))
            result = data[current_idx + 1 + c_i]
            tests.append((keys, result))
            current_idx += (c_i + 2)
        return tests

    tests = extract_tests(input_data, M)

    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If result is 'o', count of real keys in the test set must be >= K
    # If result is 'x', count of real keys in the test set must be < K
    # Note: keys are 1-indexed, so we check index (key - 1)
    
    valid_combinations_count = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for (test_keys, res) in tests
        )
    )

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()