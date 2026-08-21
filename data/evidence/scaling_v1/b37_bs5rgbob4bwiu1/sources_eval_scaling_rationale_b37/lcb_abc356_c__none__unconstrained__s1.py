import sys
from itertools import product

def solve():
    # Read N, M, K from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests
    # Each test is a tuple: (set_of_keys, result)
    # We process the flat input_data list to group tests
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # Result is at index + 1 + c_i
        result = data[index + 1 + c_i]
        
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)

    # Generate all 2^N combinations of (Real=1, Dummy=0)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    def is_valid(combo):
        # combo is a tuple of 0s and 1s corresponding to keys 1...N
        # We check if this combination satisfies all M tests
        # A test is satisfied if:
        # (R_i == 'o' AND real_keys_in_test >= K) OR (R_i == 'x' AND real_keys_in_test < K)
        
        # To avoid loops, we use all() with a generator expression
        return all(
            (sum(combo[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(combo[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )

    # Count how many combinations are valid
    # We use sum() on a generator of booleans to count True values
    ans = sum(is_valid(combo) for combo in all_combinations)
    print(ans)

if __name__ == "__main__":
    solve()