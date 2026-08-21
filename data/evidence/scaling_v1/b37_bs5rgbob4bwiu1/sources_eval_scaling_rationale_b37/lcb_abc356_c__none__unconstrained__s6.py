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
    
    # Parse tests
    # Since each test has a variable length C_i, we process the flat list
    # We use a helper function to group the input into test objects
    def parse_tests(data, index, count):
        if count == 0:
            return []
        C_i = int(data[index])
        # Keys are from index + 1 to index + C_i
        keys = [int(x) for x in data[index + 1 : index + 1 + C_i]]
        # Result is at index + 1 + C_i
        result = data[index + 1 + C_i]
        # Recurse to get the rest of the tests
        return [(keys, result)] + parse_tests(data, index + 2 + C_i, count - 1)

    tests = parse_tests(input_data, 3, M)
    
    # Generate all 2^N combinations of (real=1, dummy=0)
    # product([0, 1], repeat=N) gives all binary strings of length N
    all_combinations = product([0, 1], repeat=N)
    
    # Function to check if a specific combination satisfies all tests
    def is_valid(combo):
        # combo is a tuple where index i corresponds to key i+1
        # We check every test: 
        # For 'o', count of real keys in the set must be >= K
        # For 'x', count of real keys in the set must be < K
        return all(
            (sum(combo[key-1] for key in keys) >= K) if res == 'o' 
            else (sum(combo[key-1] for key in keys) < K)
            for keys, res in tests
        )

    # Count how many combinations are valid using a generator expression inside sum()
    ans = sum(1 for combo in all_combinations if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()