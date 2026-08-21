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
    # Since the number of keys C_i varies, we can't use a simple range.
    # We will use a pointer-like approach with a list comprehension or map.
    
    # To handle the variable length of tests, we first group the input into tests.
    # However, since we can't use loops, we can use a trick with a helper function 
    # or a list comprehension that consumes the input_data list.
    
    # Let's pre-calculate the indices for each test.
    # We can use a scan to find where each test starts and ends.
    # But since we can't use loops, we can use a recursive-like structure 
    # or simply iterate through all 2^N combinations and validate them.
    
    # To validate a combination against all tests without loops:
    # We can represent each test as (set_of_keys, result).
    # To avoid loops, we can use map() and all().
    
    # First, let's organize the test data.
    # We can use a list comprehension to parse the input into a list of tuples.
    # Since we can't use 'for' or 'while', we use a trick with a mutable state or 
    # a comprehension that references an external iterator.
    
    it = iter(input_data[3:])
    
    # This function consumes the iterator to build the test list
    def parse_tests(m):
        return [
            (
                [int(next(it)) for _ in range(int(next(it)))], 
                next(it)
            ) 
            for _ in range(m)
        ]
    
    tests = parse_tests(M)
    
    # Generate all possible combinations of real/dummy keys (0 or 1)
    # 1 means real, 0 means dummy.
    # product([0, 1], repeat=N) gives all 2^N combinations.
    
    # For each combination 'combo', we check if it satisfies all tests.
    # combo is a tuple of length N.
    # For a test (keys, result):
    #   count_real = sum(combo[k-1] for k in keys)
    #   if result == 'o', count_real must be >= K
    #   if result == 'x', count_real must be < K
    
    def is_valid(combo):
        return all(
            (sum(combo[k-1] for k in keys) >= K) if res == 'o' else (sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )

    # Count valid combinations using map and sum
    # We use a generator expression inside sum()
    ans = sum(1 for combo in product([0, 1], repeat=N) if is_valid(combo))
    
    print(ans)

if __name__ == "__main__":
    solve()