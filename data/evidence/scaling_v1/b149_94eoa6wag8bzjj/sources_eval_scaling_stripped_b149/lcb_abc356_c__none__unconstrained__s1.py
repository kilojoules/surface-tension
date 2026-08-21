import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of keys, M: number of tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests. Each test is a tuple: (set_of_keys, result)
    # We use a helper function to extract tests from the flat list
    def parse_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # Keys are from current_pos + 1 to current_pos + c_i
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += c_i + 2
        return tests

    tests = parse_tests(input_data, M)

    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if it satisfies all M tests
    # For a combination 'comb', the i-th key is real if comb[i-1] == 1
    # The number of real keys in a test is the sum of comb[key-1] for key in test_keys
    
    # We use a generator expression inside sum() to count valid combinations
    # The all() function checks if the condition holds for every test in the list
    ans = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if result == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()