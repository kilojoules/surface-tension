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

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a helper function to extract the tests from the remaining tokens.
    def parse_tests(tokens, m_count):
        tests = []
        current_pos = 0
        for _ in range(m_count):
            c_i = int(tokens[current_pos])
            # The keys are the next c_i tokens
            keys = [int(x) for x in tokens[current_pos + 1 : current_pos + 1 + c_i]]
            # The result is the token after the keys
            result = tokens[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = parse_tests(input_data[3:], M)

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys.
    # product([0, 1], repeat=N) creates an iterator of tuples.
    # We use a generator expression inside sum() to count valid combinations.
    # For each combination 'comb', we check if all M tests are satisfied.
    # A test is satisfied if:
    # (result == 'o' AND sum of real keys in test >= K) OR
    # (result == 'x' AND sum of real keys in test < K)
    
    # Note: keys in tests are 1-indexed, so we use key-1 for 0-indexed tuple access.
    
    ans = sum(
        1 for comb in product([0, 1], repeat=N)
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if result == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, result in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()