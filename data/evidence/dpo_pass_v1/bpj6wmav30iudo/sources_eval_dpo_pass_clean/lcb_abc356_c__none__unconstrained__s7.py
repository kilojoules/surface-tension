import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of strings
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Extract N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse the M tests. Since each test has a variable length C_i,
    # we process the input_data list.
    # We use a helper to slice the input_data based on the structure:
    # [C_i, A_i1, ..., A_iCi, R_i]
    
    # To avoid loops, we can pre-calculate the indices where each test starts.
    # However, since we can't use loops to build the index list, 
    # we can use a recursive-like approach via map/list comps if we knew C_i.
    # But since C_i varies, we can use a trick: 
    # we know the total number of elements is len(input_data).
    # We can use a list comprehension to extract the tests by 
    # iterating through the range of M and managing the pointer.
    # Since we can't use loops, we'll use a custom function with 
    # a reduction or a clever slice.
    
    # Actually, a simpler way to handle the variable length input without 
    # explicit loops is to use an iterator.
    it = iter(input_data[3:])
    
    # Since we can't use 'for' or 'while', we use a recursive-like 
    # structure via map/list comps. But wait, the prompt says 
    # "no function may call itself" and "no recursion".
    # We can use a list comprehension to parse the tests if we 
    # can determine the boundaries.
    
    # Let's use a different approach: 
    # Since N is small (15), we can just process the raw input_data 
    # using a helper that identifies the R_i (the 'o' or 'x') 
    # and the keys preceding it.
    
    # We can identify the positions of 'o' and 'x' in the input_data.
    # Let's find all indices i such that input_data[i] is 'o' or 'x'.
    # Then the keys for that test are the elements between the previous 'o'/'x' 
    # (plus the C_i element) and the current 'o'/'x'.
    
    # Let's use a more robust way to parse:
    # We know each test starts with C_i.
    # We can use a list comprehension to extract the tests by 
    # calculating the cumulative sum of (C_i + 2).
    # But we can't use a loop to calculate cumulative sums.
    
    # Wait, we can use `itertools.accumulate` for cumulative sums!
    from itertools import accumulate
    
    # First, we need the C_i values. But C_i is at the start of each test.
    # This is a chicken-and-egg problem. 
    # Let's use a simpler observation: R_i is always the last element of a test.
    # We can filter the input_data for 'o' and 'x' and their preceding elements.
    
    # Since we can't use loops, let's use a regex or a split 
    # if the input format allows. But the input is space-separated.
    
    # Let's use a list comprehension to find the indices of 'o' and 'x'.
    # result_indices = [i for i, val in enumerate(input_data[3:]) if val in ('o', 'x')]
    # Then we can slice the keys.
    
    res_indices = [i for i, val in enumerate(input_data[3:]) if val in ('o', 'x')]
    
    # For each result index 'ri', the keys are from (previous_ri + 2) to ri.
    # The first test starts at 0.
    # Test i keys: input_data[3 + start_of_test_i + 1 : 3 + ri]
    # where start_of_test_i is the end of the previous test.
    
    # We can use accumulate to find the start positions.
    # Let', C_vals be the C_i values.
    # C_1 is at input_data[3].
    # C_2 is at input_data[3 + C_1 + 2].
    # This is still recursive.
    
    # Let's use a different strategy: 
    # Since N is only 15, we can just check every possible combination 
    # of real/dummy keys (2^15 = 32768) against the constraints.
    # To parse the constraints without loops:
    # We can use a list comprehension to extract all tests.
    # Since we can't use loops, we'll use a helper function 
    # that processes the list using map and lambda.
    
    # Actually, we can parse the tests by simply looking for 'o' and 'x'.
    # The keys for a test are the elements between the C_i and the R_i.
    # Since A_ij are integers and R_i are letters, we can separate them.
    
    # Let's extract all R_i and the sets of keys.
    # We can use a list comprehension to find all 'o'/'x' and 
    # then slice the list relative to those positions.
    
    # Let's build the tests list:
    # Each test: (set_of_keys, result)
    # We use a list comprehension with a range based on the result indices.
    # To get the keys for test i:
    # The result is at res_indices[i].
    # The C_i is at (res_indices[i] - C_i). 
    # But we don't know C_i yet. 
    # However, we know C_i is the element right before the keys.
    # The keys are input_data[3 + (res_indices[i] - C_i + 1) : 3 + res_indices[i]]
    # where C_i = int(input_data[3 + (res_indices[i] - C_i)])
    # This is still circular.
    
    # Let's use the fact that R_i is the only non-integer in the test block.
    # We can group the input into tests by splitting the list 
    # at every 'o' or 'x'.
    
    # We can use a generator or map to process the flat list.
    # Since we can't use loops, we'll use a list comprehension 
    # that iterates through the result indices.
    
    # For result index ri:
    # The result is input_data[3 + ri]
    # The C_i is input_data[3 + ri - (C_i + 1)] ... still circular.
    # But wait: the keys are simply the integers between the 
    # previous result and the current result.
    # Test i keys: input_data[3 + (res_indices[i-1]+1 if i>0 else 0) + 1 : 3 + res_indices[i]]
    
    tests = [
        (
            {int(x) for x in input_data[3 + (res_indices[i-1]+1 if i > 0 else 0) + 1 : 3 + res_indices[i]]},
            input_data[3 + res_indices[i]]
        )
        for i in range(M)
    ]
    
    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # Each combination is a tuple of length N
    all_combos = product([0, 1], repeat=N)
    
    # A combination is valid if for all tests:
    # (count of real keys in test >= K) == (result == 'o')
    # We use sum() to count real keys. 
    # Note: keys are 1-indexed, so we check combo[key-1]
    
    valid_combos = [
        combo for combo in all_combos
        if all(
            (sum(combo[key-1] for key in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    ]
    
    # The answer is the number of valid combinations
    print(len(valid_combos))

if __name__ == "__main__":
    solve()