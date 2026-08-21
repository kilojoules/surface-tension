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
    
    # We need to parse M tests. Each test has a variable length C_i.
    # Since the input format is C_i A_{i,1} ... A_{i,C_i} R_i,
    # we process the input stream to extract these groups.
    
    tests = []
    current_pos = 3
    for _ in range(M):
        C_i = int(input_data[current_pos])
        # Keys are indices from current_pos + 1 to current_pos + C_i
        keys = [int(x) for x in input_data[current_pos + 1 : current_pos + 1 + C_i]]
        # Result is at current_pos + 1 + C_i
        result = input_data[current_pos + 1 + C_i]
        tests.append((keys, result))
        current_pos += (C_i + 2)

    # There are 2^N possible combinations of real/dummy keys.
    # We represent a combination as a tuple of 0s (dummy) and 1s (real).
    # product([0, 1], repeat=N) generates all 2^N combinations.
    
    # A combination is valid if for every test:
    # If R_i == 'o', then sum of keys in combination >= K
    # If R_i == 'x', then sum of keys in combination < K
    
    # We use a generator expression inside sum() to count valid combinations.
    # For a given combination 'p', p[i-1] is 1 if key i is real, 0 otherwise.
    
    ans = sum(
        1 for p in product([0, 1], repeat=N)
        if all(
            (sum(p[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(p[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()