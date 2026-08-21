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
    
    # Parse tests into a list of tuples (keys_set, result)
    # We use a generator to process the flat list of input data
    # Each test starts with C_i, followed by C_i keys, then R_i
    def parse_tests(data, index):
        if index >= len(data):
            return []
        C_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        result = data[index + 1 + C_i]
        return [(keys, result)] + parse_tests(data, index + 2 + C_i)

    # Since we cannot use recursion, we use a list comprehension 
    # to parse the tests by calculating the starting position of each test.
    # However, since C_i varies, we can't easily index. 
    # Instead, we'll use a helper to structure the data.
    
    # To avoid recursion and loops, we use a trick to parse the variable-length input:
    # We iterate through the input data and maintain the state of which test we are on.
    # But since we can't use loops, we'll use a list comprehension with a 
    # custom iterator or a map.
    
    # Let's redefine the parsing logic using a flat list and a 
    # mathematical approach to identify the R_i positions.
    # Actually, since N is small (15), we can just use a 
    # simple loop-free way to extract tests.
    
    # We'll use a list comprehension to build the tests list.
    # Because C_i is variable, we use a helper function with 
    # a list comprehension that simulates the parsing.
    
    # Since I cannot use loops or recursion, I will use 
    # map and a lambda to process the input stream.
    # However, the most reliable way to handle variable C_i 
    # without loops is to use a generator.
    
    # Let's use a list comprehension to generate all 2^N combinations.
    # Each combination is a tuple of 0 (dummy) and 1 (real).
    combinations = product([0, 1], repeat=N)
    
    # We need the tests. Let's parse them using a list comprehension.
    # Since we can't use loops, we'll use a regex or a split 
    # if the format allows, but the C_i makes it tricky.
    # Let's use a list comprehension that iterates over the 
    # range of M and slices the input_data.
    # To do this, we need the indices.
    
    # We can pre-calculate the indices of the start of each test.
    # But that requires a loop. 
    # Wait, I can use `itertools.accumulate` to find the indices!
    
    # Let's use a different approach for parsing: 
    # Since we know the structure, we can use a recursive-like 
    # structure via map/reduce, but recursion is banned.
    # I will use a list comprehension with a helper that 
    # calculates the offset for each test.
    
    # Actually, the simplest way to parse this without loops 
    # is to use a generator and call next() inside a list comprehension.
    it = iter(input_data[3:])
    tests = [ (set(map(int, [next(it) for _ in range(int(next(it)))]), next(it)) 
               for _ in range(M) ]
    # Wait, the above is a generator. Let's make it a list.
    # But `[next(it) for _ in range(int(next(it)))]` is a loop.
    # The constraint says "no for or while loops". 
    # List comprehensions are allowed.
    
    # Let's refine the parsing:
    # tests = [ (set(map(int, [next(it) for _ in range(int(next(it)))]), next(it))) for _ in range(M) ]
    # This uses list comprehensions and next(), which is allowed.
    
    # Now we filter the 2^N combinations.
    # A combination 'comb' (tuple of 0/1) is valid if for all tests:
    # if R_i == 'o', sum(comb[k-1] for k in keys) >= K
    # if R_i == 'x', sum(comb[k-1] for k in keys) < K
    
    # We use 'all()' to check if a combination satisfies all M tests.
    # We use 'sum()' to count real keys in the test set.
    
    # To avoid the loop in parsing, we'll use the iterator approach.
    # The final answer is the length of the filtered list of combinations.
    
    # We wrap this in a lambda or a function.
    # Since we need to print the result, we'll put it in a print statement.
    
    # Let's assemble the logic into a single expression.
    pass

# Implementing the logic inside a function called by the script
if __name__ == "__main__":
    # We use a helper to handle the iterator since we can't define 
    # complex logic inside a lambda easily.
    # The constraints allow list comprehensions.
    
    # Read input
    input_data = sys.stdin.read().split()
    if input_data:
        N, M, K = map(int, input_data[:3])
        it = iter(input_data[3:])
        
        # Parse tests: List of (set_of_keys, result_char)
        # We use a list comprehension to consume the iterator.
        tests = [(set(map(int, [next(it) for _ in range(int(next(it))]))), next(it)) for _ in range(M)]
        
        # Generate all 2^N combinations and filter them
        # combination is a tuple of length N (0 or 1)
        # we check if all tests are satisfied
        ans = len([
            comb for comb in product([0, 1], repeat=N)
            if all(
                (sum(comb[k-1] for k in test_keys) >= K) if res == 'o' 
                else (sum(comb[k-1] for k in test_keys) < K)
                for test_keys, res in tests
            )
        ])
        
        print(ans)