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
    
    # Parse tests
    # Since the number of keys C_i varies, we need a way to group the input.
    # We use a helper function to extract tests from the flat list.
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = [int(x) for x in data[index + 1 : index + 1 + c_i]]
        # Result is at index + 1 + c_i
        result = data[index + 1 + c_i]
        
        # Recursively get the rest of the tests
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of (real=1, dummy=0)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)
    
    # Validation logic for a single combination
    # combo is a tuple where combo[i] is the status of key i+1
    def is_valid(combo):
        # Check if all M tests are satisfied
        # For each test:
        # If R_i == 'o', sum of real keys in the set must be >= K
        # If R_i == 'x', sum of real keys in the set must be < K
        return all([
            (sum([combo[key-1] for key in test_keys]) >= K) if res == 'o' 
            else (sum([combo[key-1] for key in test_keys]) < K)
            for (test_keys, res) in tests
        ])

    # Count how many combinations satisfy the condition
    # We use a generator expression inside sum() to avoid creating a large list in memory
    answer = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(answer)

if __name__ == "__main__":
    solve()