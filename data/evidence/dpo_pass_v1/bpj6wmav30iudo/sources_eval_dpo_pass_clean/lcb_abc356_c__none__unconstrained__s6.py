import sys
from itertools import product

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
    # since we cannot use for/while loops.
    
    # To handle the variable length of C_i, we map the input 
    # into a structure that allows us to access the R_i values.
    # Because we cannot loop, we use a recursive-like approach 
    # via map/list comprehensions or pre-calculated indices.
    
    # First, let's extract the tests. 
    # Since C_i is given, we can't easily slice without a loop.
    # However, we can use a helper function and map.
    
    # We'll use a trick: since N is small (15), we can iterate 
    # through all 2^N combinations of real/dummy keys.
    # A combination is a tuple of 0 (dummy) and 1 (real).
    
    # To parse the tests without a loop, we use a recursive-like 
    # comprehension or simply process the input stream.
    # Given the constraints and the "no loop" rule, 
    # we can use a custom function with map/reduce or 
    # a list comprehension that consumes the input.
    
    # Since we need to parse M tests where each test has C_i elements:
    # We can use a helper to group the input.
    
    # Let's redefine the input parsing to be loop-free.
    # We use a list comprehension to identify the positions of 'o' and 'x'.
    # But the C_i values make it tricky. 
    # Let's use a recursive-style approach via map/lambda 
    # but the prompt bans recursion.
    
    # Actually, we can parse the tests by observing that 
    # each test ends with 'o' or 'x'.
    # We can use a list comprehension to find all indices of 'o' and 'x'.
    
    raw_tokens = input_data[3:]
    
    # Find indices of results (R_i)
    # R_indices = [i for i, token in enumerate(raw_tokens) if token in ('o', 'x')]
    # This is a list comprehension (allowed).
    R_indices = [i for i, token in enumerate(raw_tokens) if token in ('o', 'x')]
    
    # For each R_i, the keys are the tokens between the previous R_i and current R_i.
    # Test i keys: raw_tokens[start : R_indices[i]]
    # where start is R_indices[i-1]+1 or 0.
    
    # We can construct the tests list:
    # Each element: (set_of_keys, is_open)
    tests = [
        (
            {int(raw_tokens[j]) for j in range(
                (R_indices[i-1] + 1) if i > 0 else 0, 
                R_indices[i]
            )}, 
            raw_tokens[R_indices[i]] == 'o'
        ) 
        for i in range(M)
    ]
    
    # Generate all 2^N combinations of keys (0=dummy, 1=real)
    # Use itertools.product to avoid loops
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for all tests:
    # (sum of real keys in test set >= K) == (test result is 'o')
    # Note: keys are 1-indexed, so we use comb[key-1]
    
    valid_combinations = [
        comb for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) == is_open
            for test_keys, is_open in tests
        )
    ]
    
    # The answer is the count of valid combinations
    print(len(valid_combinations))

if __name__ == "__main__":
    solve()