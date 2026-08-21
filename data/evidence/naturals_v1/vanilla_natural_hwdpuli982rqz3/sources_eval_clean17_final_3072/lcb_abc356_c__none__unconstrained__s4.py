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
    
    # We need to parse M tests. Since C_i varies, we use an iterator.
    it = iter(input_data[3:])
    tests = []
    for _ in range(M):
        # C_i is the first element of the test
        C_i = int(next(it))
        # The next C_i elements are the keys
        keys = [int(next(it)) for _ in range(C_i)]
        # The last element is the result R_i
        result = next(it)
        tests.append((keys, result))
    
    # There are 2^N possible combinations of real/dummy keys.
    # We can represent each combination as a tuple of 0s and 1s.
    # 1 means real, 0 means dummy.
    
    # Use a generator to iterate through all 2^N combinations to save memory
    # product([0, 1], repeat=N) creates an iterator
    all_combinations = product([0, 1], repeat=N)
    
    # Define a helper to check if a combination is valid
    def is_valid(combo):
        # combo is a tuple where index i corresponds to key i+1
        for keys, result in tests:
            # Count how many keys in this test are real (1)
            # key - 1 because keys are 1-indexed and combo is 0-indexed
            count = sum(combo[k-1] for k in keys)
            
            if result == 'o':
                # Door opens if count >= K. If count < K, it's a contradiction.
                if count < K:
                    return False
            else: # result == 'x'
                # Door stays closed if count < K. If count >= K, it's a contradiction.
                if count >= K:
                    return False
        return True

    # Count valid combinations using map/filter or sum with generator
    # We use sum(1 for ...) to avoid loading everything into a list
    answer = sum(1 for combo in all_combinations if is_valid(combo))
    print(answer)

if __name__ == "__main__":
    solve()