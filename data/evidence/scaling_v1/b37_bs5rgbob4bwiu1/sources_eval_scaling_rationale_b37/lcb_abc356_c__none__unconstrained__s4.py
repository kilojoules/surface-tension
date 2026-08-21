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
    # We use a helper function to parse the flat list into a list of test tuples.
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = [int(x) for x in data[index + 1 : index + 1 + c_i]]
        # Result R_i is at index + 1 + c_i
        result = data[index + 1 + c_i]
        
        # Recursively parse the rest
        return [(keys, result)] + parse_tests(data, index + 2 + c_i, count - 1)

    tests = parse_tests(input_data, 3, M)

    # Generate all 2^N combinations of (Real=1, Dummy=0)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # Validation logic for a single combination:
    # A combination is valid if for every test:
    # - If R_i == 'o', count of real keys in the set >= K
    # - If R_i == 'x', count of real keys in the set < K
    def is_valid(combo):
        # combo is a tuple of 0s and 1s. 
        # Key j is real if combo[j-1] == 1.
        return all(
            (sum(combo[key-1] for key in keys) >= K) if res == 'o' 
            else (sum(combo[key-1] for key in keys) < K)
            for keys, res in tests
        )

    # Count how many combinations satisfy all tests
    # We use a generator expression inside sum() to count True values
    answer = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(answer)

if __name__ == "__main__":
    solve()