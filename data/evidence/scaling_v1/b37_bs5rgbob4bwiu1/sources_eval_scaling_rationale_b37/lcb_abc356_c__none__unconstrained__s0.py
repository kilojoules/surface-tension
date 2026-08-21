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
    # We use a helper function to process the flat list into a list of test tuples.
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        C_i = int(data[index])
        # The keys are from index + 1 to index + C_i
        keys = [int(x) for x in data[index + 1 : index + 1 + C_i]]
        # The result R_i is at index + 1 + C_i
        result = data[index + 1 + C_i]
        
        # Recursively parse the remaining tests
        return [(keys, result)] + parse_tests(data, index + 2 + C_i, count - 1)

    tests = parse_tests(input_data, 3, M)

    # Generate all 2^N combinations of (Real=1, Dummy=0)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # Validation logic:
    # For a combination 'comb', key j is real if comb[j-1] == 1.
    # A test (keys, res) is satisfied if:
    # - res == 'o' AND sum of real keys in the set >= K
    # - res == 'x' AND sum of real keys in the set < K
    
    def is_valid(comb):
        # Check if all tests are satisfied for this specific combination
        return all(
            (sum(comb[key-1] for key in keys) >= K) if res == 'o' 
            else (sum(comb[key-1] for key in keys) < K)
            for keys, res in tests
        )

    # Count how many combinations satisfy the is_valid predicate
    # We use a list comprehension to filter and len() to count
    ans = len([comb for comb in all_combinations if is_valid(comb)])
    
    print(ans)

if __name__ == "__main__":
    solve()