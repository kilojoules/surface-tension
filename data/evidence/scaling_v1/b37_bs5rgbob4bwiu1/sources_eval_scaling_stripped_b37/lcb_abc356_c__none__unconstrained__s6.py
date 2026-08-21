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
    
    # We need to parse the M tests. 
    # Since each test has a variable length C_i, we process the list linearly.
    tests = []
    cursor = 3
    for _ in range(M):
        C_i = int(input_data[cursor])
        # Extract the C_i keys
        keys = [int(x) for x in input_data[cursor + 1 : cursor + 1 + C_i]]
        # Extract the result R_i
        result = input_data[cursor + 1 + C_i]
        tests.append((set(keys), result))
        cursor += C_i + 2

    # A combination is a tuple of 0s (dummy) and 1s (real) of length N.
    # We use product([0, 1], repeat=N) to iterate through all 2^N possibilities.
    # For each combination, we check if it satisfies all M test conditions.
    
    # A combination is valid if for every test:
    # If R_i == 'o', then (number of real keys in the set) >= K
    # If R_i == 'x', then (number of real keys in the set) < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # combination is a tuple where index i corresponds to key i+1.
    
    ans = sum(
        1 for combo in product([0, 1], repeat=N)
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()