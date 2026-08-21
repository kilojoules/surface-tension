import sys
from itertools import product

def solve():
    # Read N, M, K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse tests
    # Each test is a tuple: (set_of_keys, result)
    # We use a helper function to consume the input list since C_i varies
    def parse_tests(data, index, count):
        tests = []
        for _ in range(count):
            C_i = int(data[index])
            keys = set(map(int, data[index + 1 : index + 1 + C_i]))
            result = data[index + 1 + C_i]
            tests.append((keys, result))
            index += C_i + 2
        return tests

    tests = parse_tests(input_data, 3, M)

    # There are 2^N possible combinations of real/dummy keys.
    # We represent a combination as a tuple of 0s and 1s of length N.
    # index i corresponds to key i+1. 1 means real, 0 means dummy.
    
    # We use product([0, 1], repeat=N) to generate all 2^N combinations.
    # For each combination, we check if it satisfies all M tests.
    # A combination is valid if for every test:
    # If R_i == 'o', then (number of real keys in set) >= K
    # If R_i == 'x', then (number of real keys in set) < K
    
    def is_valid(combo):
        # combo is a tuple like (1, 0, 1) meaning keys 1 and 3 are real.
        # To count real keys in a test set, we sum the values at (key-1).
        for keys, result in tests:
            real_count = sum(combo[k-1] for k in keys)
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Count how many combinations satisfy the condition
    ans = sum(1 for combo in product([0, 1], repeat=N) if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()