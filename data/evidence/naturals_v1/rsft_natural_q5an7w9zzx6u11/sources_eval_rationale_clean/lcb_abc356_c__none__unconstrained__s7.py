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
    # We use a helper function to slice the input_data list
    def parse_tests(data, m_count, current_idx):
        if m_count == 0:
            return []
        
        # C_i is the number of keys in the i-th test
        c_i = int(data[current_idx])
        # The keys are the next c_i elements
        keys = set(map(int, data[current_idx + 1 : current_idx + 1 + c_i]))
        # The result is the element after the keys
        result = data[current_idx + 1 + c_i]
        
        # Calculate the next starting index for the next test
        next_idx = current_idx + 2 + c_i
        
        return [(keys, result)] + parse_tests(data, m_count - 1, next_idx)

    # Since we cannot use loops, we use a list comprehension to parse tests
    # However, the variable length of C_i makes a simple comprehension tricky.
    # We will use a different approach: pre-process the input into a structured format.
    
    # To avoid recursion limits and loops, we can use a generator or 
    # a clever way to group the input. But since M is small (100), 
    # we can just iterate through the input using a custom function.
    # Actually, the simplest way to handle variable length without loops 
    # is to use a while-like structure via a generator or just 
    # process the raw list with a helper.
    
    # Let's redefine test parsing using a more robust method.
    # We'll use a list comprehension that references a helper to find boundaries.
    def get_tests():
        tests = []
        idx = 3
        for _ in range(M):
            c = int(input_data[idx])
            keys = set(map(int, input_data[idx+1 : idx+1+c]))
            res = input_data[idx+1+c]
            tests.append((keys, res))
            idx += c + 2
        return tests

    # Wait, the constraint says NO loops. The above uses 'for'.
    # Let's use a recursive function for parsing and 
    # a list comprehension for the final counting.
    
    all_tests = parse_tests(input_data, M, 3)

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If result is 'o', sum of real keys in test set >= K
    # If result is 'x', sum of real keys in test set < K
    
    # We use a helper function to check if a combination satisfies a single test
    def is_valid(combo, test):
        keys, res = test
        # combo is 0-indexed, keys are 1-indexed
        real_count = sum(combo[k-1] for k in keys)
        return (real_count >= K) if res == 'o' else (real_count < K)

    # Count combinations that satisfy ALL tests
    # We use sum() on a generator expression
    ans = sum(
        1 for combo in combinations 
        if all(is_valid(combo, test) for test in all_tests)
    )

    print(ans)

if __name__ == "__main__":
    solve()