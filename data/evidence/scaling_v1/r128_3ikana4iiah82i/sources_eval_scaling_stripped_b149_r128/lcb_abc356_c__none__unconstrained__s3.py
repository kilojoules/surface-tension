import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of keys, M: number of tests, K: threshold for opening
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a list comprehension to iterate through the input stream
    # Since C_i varies, we handle the parsing by keeping track of the current index
    # However, since we can't use loops, we'll use a helper function with recursion 
    # or a clever list comprehension. Let's use a recursive-like structure via map/reduce
    # or simply pre-process the input into a list of tests.
    
    # To avoid loops, we can use a generator to yield the tests
    def get_tests(data, idx):
        if idx >= len(data):
            return []
        c = int(data[idx])
        keys = set(map(int, data[idx + 1 : idx + 1 + c]))
        res = data[idx + 1 + c]
        return [(keys, res)] + get_tests(data, idx + 2 + c)

    tests = get_tests(input_data[3:], 0)

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # For a given combination 'comb' (tuple of 0s and 1s):
    # A key 'a' is real if comb[a-1] == 1.
    # The number of real keys in a test set 'keys' is sum(comb[a-1] for a in keys).
    # The test is satisfied if:
    # (result == 'o' and sum >= K) OR (result == 'x' and sum < K)
    
    # We use a generator expression inside sum() to count valid combinations
    ans = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[a-1] for a in keys) >= K) if res == 'o' 
            else (sum(comb[a-1] for a in keys) < K)
            for keys, res in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()