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
    
    # We need to process M tests. Since the input format for each test 
    # varies in length (C_i), we use an iterator to handle the stream.
    it = iter(input_data[3:])
    
    tests = []
    for _ in range(M):
        C_i = int(next(it))
        # Get the next C_i elements as keys
        keys = [int(next(it)) for _ in range(C_i)]
        # Get the result R_i
        result = next(it)
        tests.append((keys, result))
    
    # The number of keys N is small (up to 15).
    # We can iterate through all 2^N possible combinations of real/dummy keys.
    # A combination can be represented as a tuple of 0s and 1s of length N.
    # 1 means real, 0 means dummy.
    
    # Use map to apply the validation logic to every combination and sum the booleans.
    # product([0, 1], repeat=N) generates all 2^N combinations.
    
    def is_valid(combination):
        # combination is a tuple like (0, 1, 1, ...)
        # Note: keys in input are 1-indexed, combination is 0-indexed.
        for keys, result in tests:
            # Count how many keys in this test are real
            # We use (k-1) to map key number to index
            count_real = sum(combination[k-1] for k in keys)
            
            if result == 'o':
                # Door opens if at least K real keys
                if count_real < K:
                    return False
            else: # result == 'x'
                # Door stays closed if fewer than K real keys
                if count_real >= K:
                    return False
        return True

    # Calculate the number of valid combinations
    # We use a generator expression inside sum() to avoid loading all into memory
    answer = sum(1 for combo in product([0, 1], repeat=N) if is_valid(combo))
    
    print(answer)

if __name__ == "__main__":
    solve()