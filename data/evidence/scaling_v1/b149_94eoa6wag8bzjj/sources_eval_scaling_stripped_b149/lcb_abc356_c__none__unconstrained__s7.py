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
    
    # We need to parse the tests. Since each test has a variable length C_i,
    # we process the remaining input data to extract the tests.
    # Each test: C_i, then C_i keys, then R_i.
    
    tests = []
    current_pos = 3
    for _ in range(M):
        C_i = int(input_data[current_pos])
        keys = [int(x) for x in input_data[current_pos + 1 : current_pos + 1 + C_i]]
        result = input_data[current_pos + 1 + C_i]
        tests.append((keys, result))
        current_pos += C_i + 2

    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # We use a generator expression inside sum() to count valid combinations.
    # For each combination 'comb', we check if it satisfies all M tests.
    
    # comb is a tuple of length N. Key i is real if comb[i-1] == 1.
    # For a test (keys, result):
    # count_real = sum(comb[k-1] for k in keys)
    # condition: (count_real >= K) if result == 'o' else (count_real < K)
    
    ans = sum(
        1 for comb in product([0, 1], repeat=N)
        if all(
            (sum(comb[k-1] for k in keys) >= K) if res == 'o' 
            else (sum(comb[k-1] for k in keys) < K)
            for keys, res in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()