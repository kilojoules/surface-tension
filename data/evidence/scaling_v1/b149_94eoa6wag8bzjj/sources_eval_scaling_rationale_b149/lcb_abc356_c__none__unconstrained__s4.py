import sys
from itertools import product
from functools import reduce

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
    # We use a helper function to slice the input list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # Result is at index + 1 + c_i
        res = data[index + 1 + c_i]
        
        return [(keys, res)] + parse_tests(data, index + 2 + c_i, count - 1)

    # Since recursion is forbidden, we use a different approach to parse tests.
    # We can use a while-like structure via a generator or just pre-process the list.
    # However, the most reliable way without loops/recursion is to use a 
    # comprehension that calculates the starting index of each test.
    
    # To avoid loops/recursion for parsing, we calculate the offsets.
    # But wait, the C_i varies, making fixed-width slicing impossible.
    # Let's use a trick with a generator and next() inside a list comprehension.
    
    it = iter(input_data[3:])
    def get_tests():
        # This is a generator. We can wrap it in a list comprehension.
        # But the prompt forbids loops. A generator expression is allowed.
        # However, calling next() inside a list comprehension is the way.
        pass

    # Let's redefine parsing: since M is small (100), we can use a 
    # trick to extract tests by iterating through the list.
    # Actually, the simplest way to parse variable length inputs without 
    # for/while is to use a custom function with map/reduce or 
    # a list comprehension that manages state.
    
    # Correct approach to parse variable length:
    # We can use a helper list to store the results of the parsing.
    # Since we can't use loops, we use a list comprehension and a 
    # mutable object (like a list) to track the current index.
    
    state = {'idx': 0}
    def extract_test():
        c_i = int(input_data[3 + state['idx']])
        keys = set(map(int, input_data[4 + state['idx'] : 4 + state['idx'] + c_i]))
        res = input_data[4 + state['idx'] + c_i]
        # Update index: C_i + 2 (for C_i and R_i)
        state['idx'] += (c_i + 2)
        return (keys, res)

    # Use a list comprehension to call extract_test M times
    tests = [extract_test() for _ in range(M)]

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # Check if a combination satisfies all tests
    def is_valid(combo):
        # combo is a tuple where index i corresponds to key i+1
        # We check if all tests are satisfied
        return all(
            (sum(combo[k-1] for k in keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in keys) < K)
            for keys, res in tests
        )

    # Count valid combinations
    # Use sum() with a generator expression to count True values
    result = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(result)

if __name__ == "__main__":
    solve()