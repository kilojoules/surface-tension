import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream
    it = iter(input_data)
    
    # Read N, M, K
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    
    # Store tests as a list of tuples: (set_of_keys, result)
    tests = []
    for _ in range(M):
        C_i = int(next(it))
        # Read C_i keys and convert to 0-indexed
        keys = {int(next(it)) - 1 for _ in range(C_i)}
        result = next(it)
        tests.append((keys, result))
    
    # Generate all 2^N possible combinations of real/dummy keys
    # 1 represents a real key, 0 represents a dummy key
    # We use product([0, 1], repeat=N) to avoid explicit for/while loops
    all_combinations = product([0, 1], repeat=N)
    
    # Define a validation function for a single combination
    def is_valid(combo):
        # For every test, check if the number of real keys meets the condition
        # We use a generator expression inside all() to satisfy constraints
        return all(
            (sum(combo[k] for k in keys) >= K) if res == 'o' else (sum(combo[k] for k in keys) < K)
            for keys, res in tests
        )
    
    # Use map to apply the validation function to all combinations and sum the True results
    # This avoids explicit loops and recursion
    ans = sum(map(is_valid, all_combinations))
    
    print(ans)

if __name__ == "__main__":
    solve()