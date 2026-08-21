import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse M tests. Since each test has a variable length C_i,
    # we process the remaining tokens to extract the tests.
    # Each test: C_i, then C_i keys, then R_i.
    
    # Use a generator to handle the variable length of tests
    def get_tests(tokens):
        idx = 0
        while idx < len(tokens):
            c_i = int(tokens[idx])
            keys = [int(x) for x in tokens[idx + 1 : idx + 1 + c_i]]
            result = tokens[idx + 1 + c_i]
            yield (keys, result)
            idx += c_i + 2

    tests = list(get_tests(input_data[3:]))

    # Generate all 2^N possible combinations of keys (0 = dummy, 1 = real)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If R_i == 'o', sum of real keys in the test set >= K
    # If R_i == 'x', sum of real keys in the test set < K
    # Note: keys are 1-indexed, so we use key-1 for the combination tuple index.
    
    valid_combinations_count = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[key - 1] for key in test_keys) >= K) if result == 'o' 
            else (sum(combo[key - 1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()