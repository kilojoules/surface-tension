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
    # We use a helper to extract the tests from the remaining input stream.
    def extract_tests(data, m_count):
        tests = []
        current_pos = 0
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # Keys are from current_pos + 1 to current_pos + c_i
            keys = [int(x) for x in data[current_pos + 1 : current_pos + 1 + c_i]]
            # Result is at current_pos + 1 + c_i
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = extract_tests(input_data[3:], M)
    
    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If result is 'o', sum of keys in the combination >= K
    # If result is 'x', sum of keys in the combination < K
    # Note: keys in input are 1-indexed, so we use key-1 for 0-indexed tuple access
    
    valid_combinations = [
        comb for comb in all_combinations
        if all(
            (sum(comb[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(comb[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    ]
    
    print(len(valid_combinations))

if __name__ == "__main__":
    solve()