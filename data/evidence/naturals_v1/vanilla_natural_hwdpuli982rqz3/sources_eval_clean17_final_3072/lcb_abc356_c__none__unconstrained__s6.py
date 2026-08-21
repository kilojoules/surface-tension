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
    
    # We need to process M tests. 
    # Since the number of keys C_i varies, we can't use a simple loop with range(M).
    # We will use a pointer-like approach or a list comprehension with a helper.
    
    # To handle the variable length of input lines, we can pre-process the input 
    # into a list of tests.
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        # Extract the C_i keys
        keys = [int(x) for x in input_data[current + 1 : current + 1 + C_i]]
        # Extract the result R_i
        result = input_data[current + 1 + C_i]
        tests.append((keys, result))
        current += C_i + 2

    # The number of combinations is 2^N. Since N <= 15, 2^15 = 32,768.
    # We can iterate through all possible combinations of real/dummy keys.
    # 1 represents real, 0 represents dummy.
    
    # Generate all binary tuples of length N
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If R_i == 'o', count of real keys in the set >= K
    # If R_i == 'x', count of real keys in the set < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # Note: The keys in input are 1-indexed, so we use key-1 for 0-indexed tuple.
    
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in keys) >= K) if res == 'o' else (sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()