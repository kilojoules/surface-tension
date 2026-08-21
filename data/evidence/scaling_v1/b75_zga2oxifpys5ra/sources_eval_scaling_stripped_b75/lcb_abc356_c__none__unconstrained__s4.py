import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: total keys, M: number of tests, K: threshold for opening
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse the M tests. Since each test has a variable length C_i,
    # we process the input list using a pointer.
    tests = []
    ptr = 3
    for _ in range(M):
        c_i = int(input_data[ptr])
        # The keys are from index ptr+1 to ptr+c_i
        keys = set(map(int, input_data[ptr+1 : ptr+1+c_i]))
        # The result is at index ptr+c_i+1
        result = input_data[ptr+c_i+1]
        tests.append((keys, result))
        ptr += c_i + 2

    # Generate all 2^N possible combinations of real (1) and dummy (0) keys
    # product([0, 1], repeat=N) generates tuples of length N
    # We map these to a set or list where index i corresponds to key i+1
    all_combinations = product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If result == 'o', count of real keys in the test set must be >= K
    # If result == 'x', count of real keys in the test set must be < K
    
    # We use a generator expression inside sum() to count valid combinations
    # For a combination 'comb', the number of real keys in a test is 
    # sum(comb[key-1] for key in test_keys)
    
    valid_count = sum(
        1 for comb in all_combinations
        if all(
            (sum(comb[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(comb[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()