import sys
from itertools import product

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: number of keys, M: number of tests, K: threshold for opening
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # We need to parse M tests. Each test has a variable length C_i.
    # We use a helper to extract the tests from the flat list.
    def extract_tests(data, m_count):
        tests = []
        current_pos = 3
        for _ in range(m_count):
            c_i = int(data[current_pos])
            # The keys are from current_pos + 1 to current_pos + c_i
            keys = set(map(int, data[current_pos + 1 : current_pos + 1 + c_i]))
            # The result is the character at current_pos + 1 + c_i
            result = data[current_pos + 1 + c_i]
            tests.append((keys, result))
            current_pos += (c_i + 2)
        return tests

    tests = extract_tests(input_data, M)
    
    # Generate all 2^N possible combinations of keys (0 = dummy, 1 = real)
    # We use a generator expression inside sum() to count valid combinations.
    # For each combination 'p', we check if all M tests are satisfied.
    # A test is satisfied if:
    # (result == 'o' AND sum of real keys in set >= K) OR 
    # (result == 'x' AND sum of real keys in set < K)
    
    # We map the combination tuple 'p' to a dictionary or list for O(1) access
    # But since we can't use loops, we use a list comprehension to build the 
    # state and then check the conditions.
    
    # To avoid loops, we use a nested generator:
    # Outer: all possible key assignments (0 or 1)
    # Inner: all tests
    # We use all() to ensure every test is satisfied.
    
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