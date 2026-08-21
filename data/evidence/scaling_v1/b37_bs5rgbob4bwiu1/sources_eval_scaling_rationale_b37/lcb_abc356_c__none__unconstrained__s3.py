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

    # Parse the tests
    # Since the number of keys C_i varies, we need a way to group the input.
    # We use a helper function to process the flat list into a list of tests.
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        C_i = int(data[index])
        # Keys are from index + 1 to index + C_i
        keys = [int(x) for x in data[index + 1 : index + 1 + C_i]]
        # Result is at index + 1 + C_i
        result = data[index + 1 + C_i]
        
        # Recursive call to process remaining tests
        return [(keys, result)] + parse_tests(data, index + 2 + C_i, count - 1)

    tests = parse_tests(input_data, 3, M)

    # Generate all 2^N combinations of (real=1, dummy=0)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # Function to check if a specific combination satisfies a single test
    def is_valid_test(combo, test):
        keys, result = test
        # Count how many keys in the test are 'real' (1)
        # keys are 1-indexed, so we use k-1 for the combo tuple
        real_count = sum(combo[k-1] for k in keys)
        
        if result == 'o':
            return real_count >= K
        else:
            return real_count < K

    # Count combinations that satisfy ALL tests
    # We use a generator expression inside sum() to count True values
    answer = sum(
        1 for combo in all_combinations 
        if all(is_valid_test(combo, test) for test in tests)
    )

    print(answer)

if __name__ == "__main__":
    solve()