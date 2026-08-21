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

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a helper to extract the tests from the flat input list.
    def parse_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # Keys are from current_pos + 1 to current_pos + c_i
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            # Result is at current_pos + 1 + c_i
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = parse_tests(input_data, M)

    # Generate all 2^N combinations of keys (0: dummy, 1: real)
    # A combination is a tuple of length N.
    # We check if the combination satisfies all M tests.
    # A combination is valid if for every test:
    # - If R_i == 'o', count of real keys in the set >= K
    # - If R_i == 'x', count of real keys in the set < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # product([0, 1], repeat=N) generates all binary strings of length N.
    # For a combination 'comb', key i is real if comb[i-1] == 1.
    
    valid_combinations_count = sum(
        1 for comb in product([0, 1], repeat=N)
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )
    )

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()