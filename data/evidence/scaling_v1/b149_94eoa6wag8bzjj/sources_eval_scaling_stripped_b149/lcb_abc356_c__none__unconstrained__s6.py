import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: total keys, M: total tests, K: threshold to open
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a generator to extract the tests based on the C_i values.
    def get_tests(data, m_count):
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # The keys are from current_pos + 1 to current_pos + c_i
            keys = [int(x) for x in data[current_pos + 1 : current_pos + 1 + c_i]]
            # The result R_i is at current_pos + 1 + c_i
            result = data[current_pos + 1 + c_i]
            yield (keys, result)
            current_pos += c_i + 2

    tests = list(get_tests(input_data, M))

    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If R_i == 'o', count of real keys in the set must be >= K
    # If R_i == 'x', count of real keys in the set must be < K
    # Note: keys in input are 1-indexed, so we use key-1 for 0-indexed tuple access
    
    valid_combinations_count = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[key-1] for key in test_keys) >= K) if result == 'o' 
            else (sum(combo[key-1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()