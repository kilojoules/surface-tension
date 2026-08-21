import itertools
import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: total keys, M: number of tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # We need to parse the M tests. Since each test has a variable length C_i,
    # we process the input_data list using a custom loop or generator.
    def get_tests(data, m_count):
        idx = 3
        for _ in range(m_count):
            c_i = int(data[idx])
            # The keys are from idx + 1 to idx + c_i
            keys = set(map(int, data[idx + 1 : idx + 1 + c_i]))
            # The result is at idx + 1 + c_i
            result = data[idx + 1 + c_i]
            yield (keys, result)
            idx += c_i + 2

    tests = list(get_tests(input_data, M))

    # Generate all 2^N possible combinations of real/dummy keys.
    # A combination is represented as a tuple of 0s (dummy) and 1s (real).
    # We use a generator expression inside 'sum' to count valid combinations.
    # For a combination 'comb', the number of real keys in a test set is 
    # the sum of comb[key-1] for all keys in the set.
    
    # A combination is valid if for all tests:
    # If R_i == 'o', then count >= K
    # If R_i == 'x', then count < K
    
    valid_count = sum(
        1 for comb in itertools.product([0, 1], repeat=N)
        if all(
            (sum(comb[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(comb[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    )

    print(valid_count)

if __name__ == "__main__":
    solve()