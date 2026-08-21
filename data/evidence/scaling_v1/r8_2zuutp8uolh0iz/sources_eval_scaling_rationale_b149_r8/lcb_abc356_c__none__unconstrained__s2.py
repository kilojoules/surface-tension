import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of keys, M: number of tests, K: threshold
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a helper function to slice the input list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # Result is at index + 1 + c_i
        result = data[index + 1 + c_i]
        
        # Recursively parse the rest
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)
    
    # Validation logic:
    # For a combination 'comb' (0-indexed), key i is real if comb[i-1] == 1.
    # A test is satisfied if:
    # (R_i == 'o' and sum of real keys in test >= K) OR
    # (R_i == 'x' and sum of real keys in test < K)
    
    # We use a generator expression inside sum() to count valid combinations.
    # The 'all()' function checks if every test is satisfied for the current combination.
    ans = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(comb[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()