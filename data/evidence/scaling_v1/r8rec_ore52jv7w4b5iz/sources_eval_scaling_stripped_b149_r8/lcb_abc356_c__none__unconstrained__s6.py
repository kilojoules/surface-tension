import itertools
import sys

def solve():
    # Read N, M, K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests
    # Since the number of keys C_i varies, we need to parse the list carefully.
    # We use a generator to yield the test specifications.
    def get_tests(data):
        idx = 3
        for _ in range(M):
            C_i = int(data[idx])
            keys = set(map(int, data[idx + 1 : idx + 1 + C_i]))
            result = data[idx + 1 + C_i]
            yield (keys, result)
            idx += C_i + 2

    tests = list(get_tests(input_data))
    
    # Generate all 2^N combinations of keys being real (1) or dummy (0)
    # A combination is represented as a tuple of length N
    # We use a list comprehension to count valid combinations
    # For each combination 'comb', we check if it satisfies all M tests.
    # A test is satisfied if:
    # (result == 'o' and count of real keys in test set >= K) OR
    # (result == 'x' and count of real keys in test set < K)
    
    # To make the check efficient, we map key numbers to 0-indexed offsets
    # and use set intersections to count real keys.
    
    # We pre-calculate the set of indices for each test to avoid repeated work
    test_indices = [(set(k - 1 for k in t[0]), t[1]) for t in tests]
    
    # We iterate through all binary strings of length N
    # We use a generator expression inside sum() to count valid combinations
    ans = sum(
        1 for comb in itertools.product([0, 1], repeat=N)
        if all(
            (sum(comb[i] for i in indices) >= K) if res == 'o' 
            else (sum(comb[i] for i in indices) < K)
            for indices, res in test_indices
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()