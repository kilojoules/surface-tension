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
    
    # The input structure is irregular because C_i varies.
    # We need to parse the tests based on the C_i values.
    # We'll use a helper function to extract the tests.
    def extract_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # Keys are from current_pos + 1 to current_pos + c_i
            keys = [int(x) for x in data[current_pos + 1 : current_pos + 1 + c_i]]
            # Result is at current_pos + 1 + c_i
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = extract_tests(input_data, M)

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of tuples
    # We use a generator expression inside sum() to count valid combinations
    # A combination is valid if for all tests:
    # (sum of real keys in test >= K) == (result == 'o')
    
    ans = sum(
        1 for combo in product([0, 1], repeat=N)
        if all(
            (sum(combo[key-1] for key in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()