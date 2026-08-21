import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Extract N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests
    # Each test consists of C_i, then C_i keys, then R_i
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        keys = [int(x) - 1 for x in input_data[current + 1 : current + 1 + C_i]]
        result = input_data[current + 1 + C_i]
        tests.append((keys, result))
        current += C_i + 2

    # Generate all 2^N possible combinations of real/dummy keys
    # 1 represents a real key, 0 represents a dummy key
    # We use a generator to save memory, though N=15 is small enough for list
    all_combinations = product([0, 1], repeat=N)
    
    # Define a function to check if a combination is valid
    def is_valid(combo):
        for keys, result in tests:
            # Count how many keys in the test set are real (1)
            real_count = sum(combo[k] for k in keys)
            
            # Door opens if real_count >= K
            # If result is 'o', real_count must be >= K
            # If result is 'x', real_count must be < K
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Count valid combinations using map/filter or generator expression
    # Since we cannot use for-loops for logic, we use sum() with a generator
    answer = sum(1 for combo in all_combinations if is_valid(combo))
    
    # Print the result
    sys.stdout.write(str(answer) + '\n')

if __name__ == "__main__":
    solve()