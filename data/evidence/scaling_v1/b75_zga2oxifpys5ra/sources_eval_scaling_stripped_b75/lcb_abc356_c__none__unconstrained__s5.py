import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: total keys, M: number of tests, K: threshold for opening
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a helper to group the flat list into test specifications.
    def parse_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # Keys are from current_pos + 1 to current_pos + c_i
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            # Result is the token immediately after the keys
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = parse_tests(input_data, M)

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys.
    # product([0, 1], repeat=N) generates tuples of length N.
    # We check each combination against all M test results.
    # A combination is valid if for every test:
    # - If R_i == 'o', count of real keys in the set >= K
    # - If R_i == 'x', count of real keys in the set < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # combination is a tuple where index i corresponds to key i+1.
    
    valid_count = sum(
        1 for combination in product([0, 1], repeat=N)
        if all(
            (sum(combination[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(combination[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )
    )

    print(valid_count)

if __name__ == "__main__":
    solve()