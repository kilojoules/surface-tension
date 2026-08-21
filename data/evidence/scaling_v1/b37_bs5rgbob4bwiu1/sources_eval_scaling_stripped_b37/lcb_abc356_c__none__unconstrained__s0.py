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
    # we process the input list manually.
    tests = []
    current_pos = 3
    for _ in range(M):
        C_i = int(input_data[current_pos])
        # Keys are at indices current_pos + 1 to current_pos + C_i
        keys = set(map(int, input_data[current_pos + 1 : current_pos + 1 + C_i]))
        result = input_data[current_pos + 1 + C_i]
        tests.append((keys, result))
        current_pos += C_i + 2

    # There are 2^N possible combinations of keys being real (1) or dummy (0).
    # We use itertools.product to generate all binary strings of length N.
    # A combination is a tuple where index i corresponds to key i+1.
    
    # For each combination, check if it satisfies all M tests.
    # A combination is valid if for every test:
    # - If R_i == 'o', count of real keys in the set >= K
    # - If R_i == 'x', count of real keys in the set < K
    
    # We use a generator expression inside sum() to count valid combinations.
    ans = sum(
        1 for p in product([0, 1], repeat=N)
        if all(
            (sum(p[key-1] for key in test_keys) >= K) if res == 'o' 
            else (sum(p[key-1] for key in test_keys) < K)
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()