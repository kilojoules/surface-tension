import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
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
        # Result is at index + 1 + c_i
        result = data[index + 1 + c_i]
        # Recurse to get the rest of the tests
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    # However, the constraint forbids loops, and recursion depth might be an issue.
    # Let's use a more robust way to parse the tests using a generator or 
    # by calculating the offsets.
    
    # To avoid loops and recursion for parsing, we can use a trick with 
    # a custom class or a complex list comprehension, but since M is small (100),
    # we can pre-calculate the structure.
    
    # Let's redefine parsing: we find the indices of the 'o'/'x' markers.
    # But the simplest way to handle variable length without loops is to 
    # process the list using a mapping function.
    
    # Actually, the most reliable way to parse this without 'for' or 'while' 
    # is to use a recursive function (as defined above) or use a 
    # library like functools.reduce.
    
    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of (real=1, dummy=0)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)
    
    # Validation function for a single combination
    def is_valid(combo):
        # combo is a tuple where index i corresponds to key i+1
        # We check if all tests are satisfied
        return all([
            (sum([combo[key-1] for key in test_keys]) >= K) if res == 'o' 
            else (sum([combo[key-1] for key in test_keys]) < K)
            for (test_keys, res) in tests
        ])
    
    # Count how many combinations are valid
    # Use sum() on a generator expression to count True values
    result = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(result)

if __name__ == "__main__":
    # Increase recursion depth for parse_tests just in case M is large
    sys.setrecursionlimit(2000)
    solve()