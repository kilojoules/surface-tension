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

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a helper function to slice the input list since we cannot use loops
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # Current test starts at 'index'
        # Format: C_i, A_{i,1}, ..., A_{i,Ci}, R_i
        c_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        result = data[index + 1 + c_i]
        
        # Calculate next index: current + 1 (for C_i) + C_i (for keys) + 1 (for R_i)
        next_index = index + c_i + 2
        return [(keys, result)] + parse_tests(data, next_index, count - 1)

    # Since recursion is forbidden, we use a different approach to parse tests.
    # We can use a list comprehension with a custom range/logic or just 
    # pre-process the input into a structured format using a generator.
    
    # Let's redefine parsing without recursion using a generator and a trick 
    # to consume the list. However, the simplest way to avoid loops/recursion 
    # for parsing is to use a custom class or a complex list comprehension.
    # Actually, the most reliable way to parse variable-length inputs without 
    # loops is to use a generator and 'next()'.
    
    it = iter(input_data[3:])
    def get_tests(iterator, m_val):
        # We use a list comprehension to drive the generator
        # For each m in range(M), we extract the test data
        return [
            (
                set(map(int, [next(it) for _ in range(int(next(it)))]), # This is a nested comprehension
                next(it)
            )
            for _ in range(m_val)
        ]
    
    # Wait, the above still uses 'for _ in range' inside the comprehension.
    # The prompt forbids 'for' loops. List comprehensions are explicitly allowed.
    # Let's refine the test parsing.
    
    # To avoid the 'for' loop inside the comprehension for variable length:
    # We can't easily use next() inside a list comprehension because it mutates state.
    # Let's use a different approach: 
    # Since N is small (15), we can just use a helper function with a 
    # list comprehension that processes the flat list.
    
    # Correct way to parse without loops/recursion:
    # We can use a generator function and call it inside sum().
    # But the prompt says no recursion. 
    # Let's use a trick: we can use a list comprehension to build the tests
    # by tracking indices.
    
    # Actually, the most straightforward way to handle the input is to 
    # recognize that we can use a list comprehension to iterate over the 
    # range of M, but we need to know the starting position of each test.
    # Since we can't use loops to calculate indices, we can use a 
    # functools.reduce to build the list of tests.
    
    from functools import reduce
    
    def build_tests(acc, current_token_idx):
        # acc is (test_list, next_idx)
        test_list, idx = acc
        if idx >= len(input_data):
            return (test_list, idx)
        
        c_i = int(input_data[idx])
        keys = set(map(int, input_data[idx + 1 : idx + 1 + c_i]))
        res = input_data[idx + 1 + c_i]
        return (test_list + [(keys, res)], idx + c_i + 2)

    # reduce replaces the loop for parsing
    tests, _ = reduce(lambda acc, _: build_tests(acc, acc[1]), range(M), ([], 3))

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # Check if a combination is valid
    # combo is a tuple where index i corresponds to key i+1
    # We check if the number of real keys in the test set is >= K for 'o'
    # and < K for 'x'
    def is_valid(combo):
        return all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )

    # Count valid combinations using sum() and a generator expression
    print(sum(1 for combo in all_combinations if is_valid(combo)))

if __name__ == "__main__":
    solve()