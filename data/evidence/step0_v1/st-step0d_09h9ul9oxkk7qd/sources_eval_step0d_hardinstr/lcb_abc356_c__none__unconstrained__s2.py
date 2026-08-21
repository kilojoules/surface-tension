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
        C_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        result = data[index + 1 + C_i]
        return [(keys, result)] + parse_tests(data, index + 2 + C_i, count - 1)

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
    # We map the combination tuple to a dictionary or list for O(1) lookup.
    
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()