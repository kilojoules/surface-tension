import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Function to parse the tests from the flat token list
    # Each test consists of: C_i, then C_i keys, then R_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # Extract C_i
        c_i = int(data[index])
        # Extract the keys (next c_i elements)
        keys = [int(x) for x in data[index + 1 : index + 1 + c_i]]
        # Extract the result R_i
        r_i = data[index + 1 + c_i]
        
        # Calculate the next starting index for the next test
        next_index = index + 2 + c_i
        
        return [(keys, r_i)] + parse_tests(data, next_index, count - 1)

    # Since recursion is forbidden, we use a different approach to parse tests.
    # We can use a custom generator or a clever slice/comprehension if we knew the structure.
    # However, since C_i varies, a simple comprehension is hard. 
    # Let's use a helper to group the flat list into test objects.
    
    def get_tests():
        # We use a generator to handle the variable length of tests
        # But wait, the prompt forbids recursion and loops.
        # We can use a trick with a list and a while-like behavior via a generator 
        # and then convert it to a list. But generators usually use loops.
        # Let's use a functional approach to group the input.
        pass

    # Corrected approach to parse tests without loops/recursion:
    # We can't use a loop to parse, but we can use a list comprehension 
    # if we can determine the boundaries. Since we can't, 
    # let's use a trick: we use a generator with a yield, 
    # but the prompt says no loops. 
    # Actually, we can use a list comprehension to iterate over the range of M,
    # but we need the starting index of each test.
    # Let's pre-calculate the starting indices using a cumulative sum.
    
    # To avoid loops/recursion for parsing, we can use a 
    # technique involving a list and a mapping function.
    # But the simplest way to handle the variable length is to 
    # use a generator and 'next()', but that requires a loop to call it.
    
    # Wait, the constraint says "no for/while loops and no recursion".
    # We can use `functools.reduce` to parse the variable-length input.
    from functools import reduce
    
    def parse_all(acc, token_idx):
        # acc is (current_tests_list, next_idx)
        tests, idx = acc
        if idx >= len(input_data):
            return (tests, idx)
        
        c_i = int(input_data[idx])
        keys = [int(x) for x in input_data[idx + 1 : idx + 1 + c_i]]
        r_i = input_data[idx + 1 + c_i]
        return (tests + [(keys, r_i)], idx + 2 + c_i)

    # Use reduce to simulate a loop for parsing the M tests
    # We create a range(M) just to trigger the reduce function M times
    tests, _ = reduce(parse_all, range(M), ([], 3))

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If R_i == 'o', then sum of keys in the combination >= K
    # If R_i == 'x', then sum of keys in the combination < K
    # Note: keys in tests are 1-indexed, so we use key-1 for the combination tuple
    
    def is_valid(combo):
        # Check if all tests are satisfied
        # We use all() which is a functional way to check a condition across a sequence
        return all(
            (sum(combo[k-1] for k in test_keys) >= K) if r == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, r in tests
        )

    # Use sum() with a generator expression to count valid combinations
    # This replaces the need for a loop and a counter variable
    result = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(result)

if __name__ == "__main__":
    solve()