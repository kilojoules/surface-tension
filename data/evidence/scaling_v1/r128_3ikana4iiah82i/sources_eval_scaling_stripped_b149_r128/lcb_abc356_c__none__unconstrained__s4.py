import sys
from itertools import product

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # The input format for tests is variable length: C_i, then C_i keys, then R_i
    # We need to extract the tests into a list of (set_of_keys, result)
    # Since we cannot use loops, we use a helper function with a list comprehension
    # to parse the flat list into the structured tests.
    
    def parse_tests(data, m_count):
        # We use a list comprehension to build the tests.
        # To handle the variable length C_i, we track the current index.
        # However, since we can't use loops, we'll use a trick with a mutable 
        # state object (like a list) inside a comprehension.
        state = {'idx': 3}
        def get_test():
            c = int(data[state['idx']])
            keys = set(map(int, data[state['idx']+1 : state['idx']+1+c]))
            res = data[state['idx']+1+c]
            state['idx'] += (c + 2)
            return (keys, res)
        
        return [get_test() for _ in range(m_count)]

    tests = parse_tests(input_data, M)
    
    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)
    
    # For each combination, check if it satisfies all M tests
    # A combination is valid if for every test:
    # (sum of real keys in test set >= K) == (result == 'o')
    
    # We use a generator expression inside sum() to count valid combinations
    # Note: keys are 1-indexed, so we use key-1 for the combination tuple index
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()