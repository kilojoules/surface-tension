import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of keys, M: number of tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests into a list of tuples: (set_of_keys, result)
    # We use a helper function or a comprehension to group the variable-length test data
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is the number of keys in the current test
        c_i = int(data[index])
        # The keys are the next c_i elements
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # The result (o/x) is the element after the keys
        result = data[index + 1 + c_i]
        
        # Recursively parse the remaining tests
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # (sum of real keys in test >= K) == (result == 'o')
    # We use a generator expression inside sum() to count valid combinations
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()