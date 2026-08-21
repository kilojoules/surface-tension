import itertools

def solve():
    # Read N, M, K
    line1 = input().split()
    if not line1: return
    n, m, k = map(int, line1)
    
    # Read the M tests
    # Each test is stored as (set_of_keys, result)
    tests = [
        (set(map(int, line.split()[1:-1])), line.split()[-1])
        for _ in range(m)
    ]
    
    # Generate all 2^N possible combinations of real keys
    # A combination is represented by a set of keys that are 'real'
    # We use a generator expression inside 'sum' to count valid combinations
    # range(1 << n) generates all binary masks from 0 to 2^N - 1
    # For each mask, we determine the set of real keys: {i+1 for i in range(n) if (mask >> i) & 1}
    
    valid_count = sum(
        1 for mask in range(1 << n)
        if (
            # For every test, check if the result is consistent with the current mask
            all(
                # If result is 'o', at least K real keys must be present
                # If result is 'x', fewer than K real keys must be present
                (len(test_keys & {i + 1 for i in range(n) if (mask >> i) & 1}) >= k)
                if result == 'o' else
                (len(test_keys & {i + 1 for i in range(n) if (mask >> i) & 1}) < k)
                for test_keys, result in tests
            )
        )
    )
    
    print(valid_count)

if __name__ == "__main__":
    solve()