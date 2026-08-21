import sys
from itertools import product

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Since the input format for tests is variable length (C_i keys),
    # we need a way to parse the M tests.
    # We'll use a generator to yield the tests one by one.
    def get_tests(data, m_val):
        idx = 3
        for _ in range(m_val):
            c_i = int(data[idx])
            keys = set(map(int, data[idx + 1 : idx + 1 + c_i]))
            result = data[idx + 1 + c_i]
            yield (keys, result)
            idx += c_i + 2

    tests = list(get_tests(input_data, M))
    
    # Generate all 2^N possible combinations of keys (0 = dummy, 1 = real)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If result is 'o', count of real keys in the set >= K
    # If result is 'x', count of real keys in the set < K
    # Note: keys are 1-indexed, so we check combination[key-1]
    
    def is_valid(combo):
        for keys_set, result in tests:
            # Count how many keys in the test set are marked as real (1) in the combo
            real_count = sum(combo[k-1] for k in keys_set)
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Count valid combinations using a generator expression inside sum()
    ans = sum(1 for combo in all_combinations if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()