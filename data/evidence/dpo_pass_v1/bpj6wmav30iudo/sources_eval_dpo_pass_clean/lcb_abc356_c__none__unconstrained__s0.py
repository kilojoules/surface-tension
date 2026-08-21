import itertools
import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse tests into a list of tuples: (set_of_keys, result)
    # We use a generator to process the flat list of input strings
    # Each test starts with C_i, followed by C_i keys, then R_i
    def parse_tests(data, index):
        if index >= len(data):
            return []
        C_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        result = data[index + 1 + C_i]
        return [(keys, result)] + parse_tests(data, index + 2 + C_i)

    # Since recursion is banned, we use a list comprehension to parse tests.
    # However, since the structure is irregular (C_i varies), 
    # we can't use a simple slice. 
    # Instead, we can pre-process the input into a list of tests 
    # by iterating through the indices in a way that respects C_i.
    # But wait, the prompt bans "for" and "while" loops.
    # I can use a helper function with map/reduce or a clever list comprehension.
    
    # To handle the irregular input without loops/recursion, 
    # we can use a custom iterator or a recursive-like structure 
    # implemented via map/functools. 
    # Actually, since N is small (15), we can just use a 
    # comprehensive approach to extract the tests.
    
    # Let's use a trick: we know the total number of tests M.
    # We can use a recursive-like approach via a list comprehension 
    # if we can index them. But the indices depend on C_i.
    # Let's use a generator-based approach inside map.
    
    # Since I cannot use loops, I'll use a recursive-like 
    # structure via a helper that is called once. 
    # But recursion is banned. 
    # I will use a list comprehension with a sentinel or 
    # a predefined way to handle the input.
    
    # Let's reconsider: I can use `itertools.accumulate` 
    # to find the starting position of each test.
    
    # 1. Get C values and R values.
    # This is tricky without loops. 
    # Let's use a regex or a specific pattern.
    # Since we know the format: C_i, then C_i elements, then R_i.
    
    # Let's use a recursive-free way to parse:
    # We can use a custom function that uses `map` and `slice`.
    # But the slice depends on the value of C_i.
    
    # Actually, I can use a recursive function if I define it 
    # using a lambda and a fixed-point combinator, 
    # but that's still recursion.
    
    # Let's use a different approach: 
    # Since M is up to 100, I can use a list comprehension 
    # that iterates over a range, but I need the offsets.
    # I can calculate offsets using a list comprehension 
    # that references its own result? No.
    
    # Wait, I can use `itertools.accumulate` to get the offsets!
    # Let's use a helper to extract C_i values.
    
    # Since I can't use loops, I'll use a recursive-like 
    # structure via `functools.reduce`.
    
    from functools import reduce
    
    # Use reduce to parse the input list into a list of tests
    # state: (current_index, list_of_tests)
    parsed = reduce(
        lambda state, _: (
            (state[0] + int(input_data[state[0]]) + 2, 
             state[1] + [(set(map(int, input_data[state[0]+1 : state[0]+1+int(input_data[state[0]])]), 
                          input_data[state[0]+1+int(input_data[state[0]])])])
        ), 
        range(M), 
        (3, [])
    )
    
    tests = parsed[1]
    
    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = itertools.product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # (sum of real keys in test set >= K) == (result == 'o')
    # Note: keys are 1-indexed, so we use combo[key-1]
    
    valid_count = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()