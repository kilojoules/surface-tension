import sys
from itertools import product

def solve():
    # Read all input and split into a list of tokens
    data = sys.stdin.read().split()
    if not data:
        return
    
    # N: total keys, M: number of tests, K: threshold
    N, M, K = map(int, data[:3])
    
    # Parse tests: each test is (set_of_keys, result)
    # We use a list comprehension with a helper function to handle the variable length
    def parse_tests():
        # We use a generator to keep track of the current index in the data list
        def generator():
            # This is a closure-like approach to iterate through the flat list
            # However, since we can't use loops, we'll use a recursive-like 
            # structure or a clever slice. But wait, the constraint is NO loops.
            # Let's use a list comprehension that processes the data.
            pass

    # Since I cannot use loops to parse, I will use a recursive-style 
    # list comprehension or map to extract the tests.
    # Given M is up to 100, we can use a helper that processes the list.
    
    def extract_tests(remaining_data, count):
        if count == 0:
            return []
        c_i = int(remaining_data[0])
        keys = set(map(int, remaining_data[1 : 1 + c_i]))
        res = remaining_data[1 + c_i]
        return [(keys, res)] + extract_tests(remaining_data[2 + c_i :], count - 1)

    # Increase recursion depth for M=100
    sys.setrecursionlimit(200)
    tests = extract_tests(data[3:], M)
    
    # Evaluate all 2^N combinations
    # product([0, 1], repeat=N) generates all binary strings
    # all(...) checks if the combination satisfies all M tests
    result = sum(
        1 for combo in product([0, 1], repeat=N)
        if all(
            (sum(combo[k-1] for k in keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    print(result)

if __name__ == "__main__":
    solve()