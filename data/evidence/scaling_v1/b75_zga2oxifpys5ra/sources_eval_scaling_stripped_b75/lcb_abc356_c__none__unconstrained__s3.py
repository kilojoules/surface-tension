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
    # We use a helper to slice the input_data list
    def parse_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = parse_tests(input_data, M)
    
    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) generates all binary strings of length N
    # We map each combination to a set of indices that are 'real'
    # Combination is represented as a tuple where index i is key i+1
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If result == 'o', count of real keys in the test set must be >= K
    # If result == 'x', count of real keys in the test set must be < K
    
    # We use a generator expression inside sum() to count valid combinations
    # For a given combination 'comb', the number of real keys in a test is:
    # sum(comb[key-1] for key in test_keys)
    
    valid_count = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()