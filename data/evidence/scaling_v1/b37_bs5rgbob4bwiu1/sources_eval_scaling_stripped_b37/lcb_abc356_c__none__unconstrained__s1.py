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
    # Each test has a variable length C_i, followed by C_i keys, then a result R_i.
    # Since the input format is slightly irregular for simple mapping, 
    # we process the remaining input stream manually.
    
    cursor = 3
    tests = []
    for _ in range(M):
        C_i = int(input_data[cursor])
        # Keys are indices 1 to N, we store them in a set for O(1) lookup
        keys = set(map(int, input_data[cursor + 1 : cursor + 1 + C_i]))
        result = input_data[cursor + 1 + C_i]
        tests.append((keys, result))
        cursor += C_i + 2

    # There are 2^N possible combinations of keys being real (1) or dummy (0).
    # We represent each combination as a tuple of length N.
    # combination[i] == 1 means key (i+1) is real.
    
    # We use a generator expression inside sum() to count valid combinations.
    # For each combination, we check if it satisfies all M tests.
    # A test is satisfied if:
    # (Number of real keys in the test set >= K) == (Result is 'o')
    
    ans = sum(
        1 for combo in product([0, 1], repeat=N)
        if all(
            (sum(combo[key-1] for key in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()