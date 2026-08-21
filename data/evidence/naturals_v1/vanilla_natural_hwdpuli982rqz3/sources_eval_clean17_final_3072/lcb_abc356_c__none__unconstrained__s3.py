import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse M tests. Since each test has a variable length C_i,
    # we use a pointer-like approach with a list comprehension or loop.
    # However, since we can't use while loops, we'll pre-process the data.
    
    # To handle the variable length of tests without while/for loops for parsing,
    # we can use a recursive-like structure or simply map the input into a list
    # and then use a helper to group them. 
    # Given the constraints on loops, I will use a list comprehension to 
    # identify the indices of the 'o'/'x' markers.
    
    # Let's find the positions of 'o' and 'x'
    markers = [i for i, val in enumerate(input_data[3:], 3) if val in ('o', 'x')]
    
    # Now we can extract each test.
    # Test i starts at markers[i-1] - C_i and ends at markers[i-1].
    # But we need C_i first.
    
    # Let's use a different approach to parse: 
    # Since we know the structure, we can use a custom function with map/reduce 
    # or just a comprehension if we can calculate the offsets.
    
    # To avoid loops, we can use a technique to group the input.
    # We know that each test starts with C_i, then C_i numbers, then R_i.
    # We can use a recursive function to parse the list.
    
    def parse_tests(data, idx):
        if idx >= len(data):
            return []
        c_i = int(data[idx])
        keys = [int(x) for x in data[idx + 1 : idx + 1 + c_i]]
        result = data[idx + 1 + c_i]
        return [(keys, result)] + parse_tests(data, idx + 2 + c_i)

    # Python's recursion limit is 1000, M is 100, so this is safe.
    tests = parse_tests(input_data, 3)
    
    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # We use product([0, 1], repeat=N)
    all_combinations = product([0, 1], repeat=N)
    
    # For each combination, check if it satisfies all M tests.
    # A combination 'comb' is a tuple of length N.
    # Key j is real if comb[j-1] == 1.
    
    def is_valid(comb):
        # Check every test (keys, result)
        # We use all() to ensure every test is satisfied.
        return all(
            (sum(comb[k-1] for k in keys) >= K) if res == 'o' else (sum(comb[k-1] for k in keys) < K)
            for keys, res in tests
        )

    # Count valid combinations using sum() and a generator
    answer = sum(1 for comb in all_combinations if is_valid(comb))
    print(answer)

if __name__ == "__main__":
    solve()