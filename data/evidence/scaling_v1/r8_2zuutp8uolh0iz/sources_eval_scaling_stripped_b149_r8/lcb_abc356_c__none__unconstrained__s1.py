import itertools
import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N: total keys, M: number of tests, K: minimum real keys to open
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    # Parse the tests. Each test is a tuple (set_of_keys, result)
    # We use a generator to process the flat input list into test structures
    def parse_tests(data, m_count):
        current_idx = 3
        for _ in range(m_count):
            c_i = int(data[current_idx])
            keys = set(map(int, data[current_idx + 1 : current_idx + 1 + c_i]))
            result = data[current_idx + 1 + c_i]
            current_idx += c_i + 2
            yield (keys, result)

    tests = list(parse_tests(input_data, M))
    
    # Generate all 2^N possible combinations of real/dummy keys
    # A combination is represented as a tuple of 0s (dummy) and 1s (real)
    # index i corresponds to key i+1
    all_combinations = itertools.product([0, 1], repeat=N)
    
    # A combination is valid if for every test:
    # If result is 'o', count of real keys in the test set must be >= K
    # If result is 'x', count of real keys in the test set must be < K
    def is_valid(combo):
        # Create a mapping for quick lookup: key -> is_real
        # Since N is small, we can use a list comprehension to count real keys
        # for each test.
        return all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for (test_keys, res) in tests
        )

    # Count how many combinations satisfy all test conditions
    result = sum(1 for combo in all_combinations if is_valid(combo))
    print(result)

if __name__ == "__main__":
    solve()