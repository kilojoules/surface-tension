import sys
from itertools import product

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse tests
    # Each test consists of C_i, then C_i keys, then R_i
    tests = []
    current = 3
    for _ in range(M):
        C_i = int(input_data[current])
        keys = [int(x) - 1 for x in input_data[current + 1 : current + 1 + C_i]]
        result = input_data[current + 1 + C_i]
        tests.append((keys, result))
        current += 2 + C_i

    # Generate all 2^N combinations of real/dummy keys
    # 1 represents real, 0 represents dummy
    combinations = product([0, 1], repeat=N)
    
    # Count combinations that satisfy all test constraints
    # A combination is valid if for every test:
    # If R_i == 'o', sum of keys in test >= K
    # If R_i == 'x', sum of keys in test < K
    
    valid_count = sum(
        1 for combo in combinations 
        if all(
            (sum(combo[k] for k in keys) >= K) if res == 'o' else (sum(combo[k] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()