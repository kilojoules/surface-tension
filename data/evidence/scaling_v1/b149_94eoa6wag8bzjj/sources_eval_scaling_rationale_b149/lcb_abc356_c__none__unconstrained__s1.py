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
    # We use a generator to parse the variable-length test lines
    def parse_tests(data, index):
        if index >= len(data):
            return []
        
        C_i = int(data[index])
        # The keys are from index+1 to index+C_i
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        # The result R_i is at index + 1 + C_i
        result = data[index + 1 + C_i]
        
        return [(keys, result)] + parse_tests(data, index + 2 + C_i)

    # Since recursion depth might be an issue for very large M (though M=100 is fine),
    # but the constraint forbids loops, we use a list comprehension to parse tests.
    # However, the variable length makes a simple comprehension tricky.
    # Let's use a different approach to parse: 
    # We can pre-calculate the starting positions of each test.
    
    def get_test_bounds(data):
        # We need to find the start of each test block.
        # Since we can't use loops, we use a helper function with a list comprehension
        # to find indices. But wait, the constraint allows function definitions.
        # Let's use a trick with a list comprehension and a mutable object to track index.
        state = {'idx': 3}
        def next_test():
            start = state['idx']
            c = int(data[start])
            state['idx'] = start + c + 2
            return (set(map(int, data[start+1 : start+1+c])), data[start+1+c])
        
        return [next_test() for _ in range(M)]

    tests = get_test_bounds(input_data)

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If R_i == 'o', sum of real keys in the test set >= K
    # If R_i == 'x', sum of real keys in the test set < K
    # Note: keys are 1-indexed, so we use comb[key-1]
    
    valid_combinations_count = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(comb[k-1] for k in test_keys) < K)
            for (test_keys, res) in tests
        )
    )

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()