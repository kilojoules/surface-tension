import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse tests
    # Each test is stored as (set_of_keys, result)
    # We use a pointer-like approach with a list comprehension to parse the variable-length tests
    def parse_tests(data, index, count):
        if count == 0:
            return []
        # C_i is at data[index]
        # Keys are from index + 1 to index + C_i
        # Result is at index + C_i + 1
        c_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        res = data[index + 1 + c_i]
        return [(keys, res)] + parse_tests(data, index + 2 + c_i, count - 1)

    # Since recursion depth is limited and we can't use loops, 
    # we use a more robust way to parse the tests using a comprehension.
    # We first identify the starting index of each test.
    def get_test_indices(data, index, count, acc):
        if count == 0:
            return acc
        c_i = int(data[index])
        return get_test_indices(data, index + 2 + c_i, count - 1, acc + [index])

    # To avoid recursion and loops, we can use a trick with a list to 
    # iteratively build the test list, but the constraints allow 
    # a simpler approach: since M is small (100), we can use a 
    # comprehension that references a helper function.
    
    # Actually, the most reliable way to parse this without loops 
    # is to pre-process the input into a format we can map over.
    # Let's use a generator-based approach to consume the input.
    it = iter(input_data)
    next(it); next(it); next(it) # Skip N, M, K
    
    def consume_tests(iterator, m):
        # Use a list comprehension to drive the consumption of the iterator
        # We create a dummy range to trigger the side-effect of next()
        return [
            (set(map(int, [next(iterator)] * 0 + [next(it) for _ in range(int(next(it))])), next(it)))
            for _ in range(m)
        ]
    
    # The above consume_tests uses a loop inside the comprehension. 
    # To be strictly loop-free, we must handle the variable length differently.
    # Let's redefine the parsing logic.
    
    # Re-reading input to use a different strategy
    raw = input_data[3:]
    
    # We can find the boundaries of each test by observing that 
    # each test starts with C_i and ends with R_i.
    # Since we cannot use loops, we use a recursive-like structure 
    # via a list comprehension and a helper.
    
    def build_tests(data):
        if not data:
            return []
        c_i = int(data[0])
        keys = set(map(int, data[1 : 1 + c_i]))
        res = data[1 + c_i]
        return [(keys, res)] + build_tests(data[2 + c_i :])

    tests = build_tests(raw)
    
    # Generate all 2^N combinations (0 = dummy, 1 = real)
    # Each combination is a tuple of length N
    all_combinations = itertools.product([0, 1], repeat=N)
    
    # Validation logic:
    # For a combination 'comb', key i is real if comb[i-1] == 1.
    # A test (keys, res) is satisfied if:
    # (res == 'o' and sum(comb[k-1] for k in keys) >= K) OR
    # (res == 'x' and sum(comb[k-1] for k in keys) < K)
    
    # We use a generator expression inside sum() to count valid combinations.
    ans = sum(
        1 for comb in all_combinations
        if all(
            (res == 'o' and sum(comb[k-1] for k in keys) >= K) or
            (res == 'x' and sum(comb[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    # Increase recursion depth for build_tests
    sys.setrecursionlimit(2000)
    solve()