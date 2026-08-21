import itertools
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse M tests. Each test has a variable length C_i.
    # We use a generator to extract the tests from the flat input list.
    def get_tests(data, m_count):
        idx = 3
        for _ in range(m_count):
            c_i = int(data[idx])
            keys = set(map(int, data[idx + 1 : idx + 1 + c_i]))
            result = data[idx + 1 + c_i]
            yield (keys, result)
            idx += c_i + 2

    tests = list(get_tests(input_data, M))

    # Generate all 2^N possible combinations of real/dummy keys.
    # A combination is represented as a tuple of 0s and 1s.
    # 1 means the key is real, 0 means it is a dummy.
    # We use a helper to check if a combination is valid.
    def is_valid(combo):
        # combo is a tuple where index i corresponds to key i+1
        # For each test, count how many keys in the test set are 'real' (1)
        for keys, result in tests:
            # Count real keys in the current test set
            # keys contains 1-based indices, so we check combo[k-1]
            real_count = sum(combo[k-1] for k in keys)
            
            if result == 'o':
                # Door opens if at least K real keys are inserted
                if real_count < K:
                    return False
            else: # result == 'x'
                # Door does not open if fewer than K real keys are inserted
                if real_count >= K:
                    return False
        return True

    # Use itertools.product to generate all 2^N combinations
    # and count how many satisfy the is_valid condition.
    ans = sum(1 for combo in itertools.product([0, 1], repeat=N) if is_valid(combo))
    
    print(ans)

if __name__ == "__main__":
    solve()