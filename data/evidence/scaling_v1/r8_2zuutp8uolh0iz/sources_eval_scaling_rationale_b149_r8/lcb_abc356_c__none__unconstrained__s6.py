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
    
    # Parse the tests
    # Since the number of keys C_i varies, we need a way to group the input.
    # We use a helper function to extract tests from the flat list.
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = [int(x) for x in data[index + 1 : index + 1 + c_i]]
        # Result R_i is at index + 1 + c_i
        r_i = data[index + 1 + c_i]
        
        return [(keys, r_i)] + parse_tests(data, index + 2 + c_i, count - 1)

    # Using a list comprehension to parse tests instead of recursion to avoid depth limits
    # and to strictly follow the "no for loop" rule while handling variable lengths.
    # However, since we can't use for loops, we can use a trick with a generator 
    # or a mapping function. Let's use a more robust approach to parse the tests.
    
    # We can use a generator-like approach with map and a custom reducer or 
    # simply process the flat list by tracking indices.
    # Given the constraints and the "no loop" rule, we can use a recursive-like 
    # structure via map/reduce or a clever list comprehension.
    
    # Let's redefine the test parsing using a helper that finds the boundaries.
    def get_tests():
        # We use a list to store the current index and update it.
        # Since we can't use loops, we use a recursive-style list comprehension.
        # But wait, the simplest way to handle the variable C_i is to 
        # pre-process the input into a format we can handle.
        
        # We'll use a helper function and map to iterate through the tests.
        # Because we can't use 'for', we use a recursive function to gather the tests.
        def recurse(idx, remaining):
            if remaining <= 0:
                return []
            c_i = int(input_data[idx])
            keys = [int(x) for x in input_data[idx + 1 : idx + 1 + c_i]]
            res = input_data[idx + 1 + c_i]
            return [(keys, res)] + recurse(idx + 2 + c_i, remaining - 1)
        
        return recurse(3, M)

    tests = get_tests()

    # Generate all 2^N combinations of real (1) and dummy (0) keys.
    # Each combination is a tuple of length N.
    all_combinations = product([0, 1], repeat=N)

    # Check if a combination satisfies a single test
    def is_valid(combo, test):
        keys, result = test
        # Count how many keys in the test are real. 
        # Note: keys are 1-indexed, combo is 0-indexed.
        real_count = sum(combo[k-1] for k in keys)
        if result == 'o':
            return real_count >= K
        else:
            return real_count < K

    # A combination is valid if it satisfies ALL tests.
    # We use 'all()' to check the condition across all tests.
    # We use a generator expression inside sum() to count valid combinations.
    ans = sum(
        1 for combo in all_combinations 
        if all(is_valid(combo, t) for t in tests)
    )
    
    print(ans)

if __name__ == "__main__":
    solve()