import sys
from itertools import product

def solve():
    # Read N, M, K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse M tests. Since each test has a variable length C_i,
    # we process the input list using a pointer/index.
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        # Extract the C_i keys
        keys = [int(x) for x in input_data[current + 1 : current + 1 + C_i]]
        # Extract the result R_i
        result = input_data[current + 1 + C_i]
        tests.append((set(keys), result))
        current += C_i + 2

    # There are 2^N possible combinations of real/dummy keys.
    # We can represent each combination as a tuple of 0s (dummy) and 1s (real).
    # product([0, 1], repeat=N) generates all 2^N combinations.
    
    # A combination is valid if for every test:
    # If R_i == 'o', then (number of real keys in the set) >= K
    # If R_i == 'x', then (number of real keys in the set) < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # For a combination 'p', p[i-1] is 1 if key i is real, 0 otherwise.
    
    ans = sum(
        1 for p in product([0, 1], repeat=N)
        if all(
            sum(p[key-1] for key in test_keys) >= K if res == 'o' 
            else sum(p[key-1] for key in test_keys) < K
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()