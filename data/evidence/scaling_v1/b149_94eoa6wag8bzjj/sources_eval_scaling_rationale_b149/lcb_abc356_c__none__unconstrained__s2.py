import itertools
import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a generator to parse the variable-length test lines
    def parse_tests(data, index):
        if index >= len(data):
            return []
        
        # C_i is the number of keys
        c_i = int(data[index])
        # The next c_i elements are the keys
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # The element after the keys is the result R_i
        result = data[index + 1 + c_i]
        
        return [(keys, result)] + parse_tests(data, index + 2 + c_i)

    # Since recursion depth is limited and we must avoid loops, 
    # we can use a clever slice-based approach to group tests.
    # However, a more robust way to handle the variable length without loops 
    # is to pre-process the input into a structured format.
    
    # To avoid the recursion limit and loops, we use a list comprehension 
    # to find the starting indices of each test.
    # But since C_i varies, we can't use a fixed step.
    # Let's use a helper function with a comprehension to extract tests.
    
    def get_tests():
        # We use a temporary list to store the results of the parsing
        # Because we cannot use while loops, we use a recursive-like 
        # structure inside a list comprehension or map.
        # Given M is small (100), we can use a trick with a dictionary 
        # or a custom class to maintain state, but that's overkill.
        # Let's use a more direct approach: 
        # We know the structure: C_i, then C_i keys, then R_i.
        
        # We can use a generator to yield the tests and convert to a list.
        def gen_tests(tokens):
            it = iter(tokens)
            try:
                while True:
                    c_i = int(next(it))
                    keys = {next(it) for _ in range(c_i)}
                    res = next(it)
                    yield (keys, res)
            except StopIteration:
                pass
        
        # Wait, the prompt says no loops. 'while' and 'for' are loops.
        # I must use comprehensions and map/filter.
        # To handle variable length, I will process the tokens by 
        # calculating the offsets.
        pass

    # Correct approach to parse variable length without loops:
    # Use a recursive function to build the test list.
    def build_test_list(tokens):
        if not tokens:
            return []
        c_i = int(tokens[0])
        keys = set(tokens[1 : 1 + c_i])
        res = tokens[1 + c_i]
        return [(keys, res)] + build_test_list(tokens[2 + c_i :])

    tests = build_test_list(input_data[3:])

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = itertools.product([0, 1], repeat=N)

    # Validation logic:
    # For a combination 'comb', key i is real if comb[i-1] == 1.
    # A test (keys, res) is satisfied if:
    # (res == 'o' and sum(comb[k-1] for k in keys) >= K) OR
    # (res == 'x' and sum(comb[k-1] for k in keys) < K)
    
    # We use a nested comprehension to count valid combinations.
    # The outer layer iterates over all 2^N possibilities.
    # The inner layer checks if all M tests are satisfied.
    
    valid_count = sum(
        1 for comb in all_combinations
        if all(
            (res == 'o' and sum(comb[int(k)-1] for k in keys) >= K) or
            (res == 'x' and sum(comb[int(k)-1] for k in keys) < K)
            for keys, res in tests
        )
    )

    print(valid_count)

# Increase recursion depth for build_test_list just in case M is large
sys.setrecursionlimit(2000)
solve()