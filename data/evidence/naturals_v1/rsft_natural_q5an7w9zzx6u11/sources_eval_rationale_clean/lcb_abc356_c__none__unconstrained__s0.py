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
    
    # Parse tests into a list of tuples: (set_of_keys, result)
    # We use a helper function to slice the input list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        C_i = int(data[index])
        # Keys are from index + 1 to index + C_i
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        # Result is at index + 1 + C_i
        res = data[index + 1 + C_i]
        
        # Recursively get the rest of the tests
        return [(keys, res)] + parse_tests(data, index + 2 + C_i, count - 1)

    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of (real=1, dummy=0)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If R_i == 'o', sum of real keys in test >= K
    # If R_i == 'x', sum of real keys in test < K
    # We use a generator expression inside sum() to count valid combinations
    
    # To avoid loops, we use a nested comprehension:
    # For each combination, we check if all tests are satisfied.
    # The check for a single test is:
    # (res == 'o' and sum(keys) >= K) or (res == 'x' and sum(keys) < K)
    
    # Since we cannot use 'for' loops, we use map/filter or comprehensions.
    # We represent the combination as a dictionary or list for easy lookup.
    
    ans = sum(
        1 for combo in all_combinations
        if all(
            (res == 'o' and sum(combo[k-1] for k in keys) >= K) or
            (res == 'x' and sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()