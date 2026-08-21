import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse tests
    # Since the number of keys C_i varies, we need a custom parser
    # to extract the tests from the flat list.
    def get_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += c_i + 2
        return tests

    tests = get_tests(input_data, M)

    # Generate all 2^N combinations of real/dummy keys
    # A combination is represented as a tuple of 0s (dummy) and 1s (real)
    # index 0 corresponds to key 1, index 1 to key 2, etc.
    all_combinations = itertools.product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If result is 'o', sum of real keys in the set must be >= K
    # If result is 'x', sum of real keys in the set must be < K
    
    # We use a generator expression inside sum() to count valid combinations
    # For each combination 'comb', we check if all tests are satisfied
    # comb[key-1] gives the status of key 'key'
    
    valid_count = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if result == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )

    print(valid_count)

if __name__ == "__main__":
    solve()