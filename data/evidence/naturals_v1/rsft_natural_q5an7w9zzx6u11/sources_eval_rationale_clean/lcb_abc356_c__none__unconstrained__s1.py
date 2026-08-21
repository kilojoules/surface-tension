import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse tests into a list of tuples: (set_of_keys, result)
    # We use a helper function to slice the input list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # Result is at index + 1 + c_i
        res = data[index + 1 + c_i]
        
        # Recursively get the rest of the tests
        return [(keys, res)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = itertools.product([0, 1], repeat=N)
    
    # Validation logic for a single combination
    # combo is a tuple where index i corresponds to key i+1
    def is_valid(combo):
        # A combination is valid if ALL tests are satisfied
        return all(
            # For each test:
            # If result is 'o', count of real keys in the set must be >= K
            # If result is 'x', count of real keys in the set must be < K
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )

    # Count how many combinations satisfy the condition
    # We use map() and sum() to avoid explicit for-loops
    result = sum(map(is_valid, all_combinations))
    print(result)

if __name__ == "__main__":
    solve()