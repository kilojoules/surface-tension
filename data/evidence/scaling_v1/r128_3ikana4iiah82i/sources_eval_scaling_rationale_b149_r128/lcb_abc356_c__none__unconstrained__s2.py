import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of keys, M: number of tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests. Each test is stored as (set_of_keys, result)
    # We use a helper function to slice the input_data list based on the 
    # variable length C_i of each test.
    
    # To avoid loops, we pre-calculate the starting indices of each test.
    # However, since we cannot use loops to build the index list, 
    # we can use a recursive-like structure via a list comprehension 
    # if we know the structure, but the C_i varies.
    # Actually, the most reliable way to parse this without loops 
    # is to process the tokens using a custom function and 
    # slicing, but since we can't use while, we use a 
    # trick with a list comprehension and a mutable state (list) 
    # to track the current pointer.
    
    ptr = [3]
    def get_test():
        c = int(input_data[ptr[0]])
        keys = set(map(int, input_data[ptr[0]+1 : ptr[0]+1+c]))
        res = input_data[ptr[0]+1+c]
        ptr[0] += (c + 2)
        return (keys, res)

    # Generate the tests list using a list comprehension
    tests = [get_test() for _ in range(M)]

    # Generate all 2^N combinations of (real=1, dummy=0)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # (count of real keys in test >= K) == (result == 'o')
    # We use a generator expression inside sum() to count valid combinations.
    
    # Note: keys in tests are 1-indexed, so we use key-1 for 0-indexing.
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()