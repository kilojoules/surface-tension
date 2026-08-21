import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: total keys, M: number of tests, K: minimum real keys to open
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a generator to extract the tests based on the C_i values.
    def get_tests(data, m_count):
        ptr = 3
        for _ in range(m_count):
            c_i = int(data[ptr])
            # Keys are from ptr+1 to ptr+c_i
            keys = [int(x) for x in data[ptr+1 : ptr+1+c_i]]
            # Result is at ptr+1+c_i
            result = data[ptr+1+c_i]
            yield (keys, result)
            ptr += c_i + 2

    tests = list(get_tests(input_data, M))

    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # product([0, 1], repeat=N) creates an iterator of all binary strings of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If result is 'o', count of real keys in the set must be >= K
    # If result is 'x', count of real keys in the set must be < K
    # Note: keys in input are 1-indexed, so we use key-1 for 0-indexed tuple access
    
    def is_valid(combo):
        for keys, result in tests:
            # Count how many of the keys in this test are 'real' (1) in the current combo
            real_count = sum(combo[k-1] for k in keys)
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Use a generator expression inside sum() to count valid combinations
    # This avoids explicit loops and follows the constraint of providing a complete solution
    ans = sum(1 for combo in all_combinations if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()