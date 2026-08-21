import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: total keys, M: total tests, K: required real keys
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests. Since the number of keys C_i varies, 
    # we need a way to group the input.
    # We use a generator to process the flat list of tokens.
    def get_tests(data):
        it = iter(data[3:])
        for _ in range(M):
            # C_i is the first element of the test group
            c_i = int(next(it))
            # The next c_i elements are the key indices
            keys = [int(next(it)) for _ in range(c_i)]
            # The element after the keys is the result R_i
            result = next(it)
            yield (set(keys), result)

    tests = list(get_tests(input_data))

    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # product([0, 1], repeat=N) creates an iterator of tuples
    all_combinations = product([0, 1], repeat=N)

    # For a combination to be valid, it must satisfy all M tests.
    # A combination is represented as a tuple where index i corresponds to key i+1.
    # We use a helper function to check if a combination satisfies the door logic.
    def is_valid(combo):
        # Create a set of indices (0-indexed) that are 'real' in this combination
        real_keys = {i for i, val in enumerate(combo) if val == 1}
        
        # Check every test:
        # For each test, calculate how many inserted keys are in the real_keys set.
        # Note: A_{i,j} are 1-indexed, so we subtract 1 to match 0-indexed real_keys.
        # However, it's easier to just map the test keys to 0-indexed and use intersection.
        
        # We use all() to ensure every test result is consistent with the combination.
        return all(
            (len([k for k in test_keys if (k-1) in real_keys]) >= K) == (res == 'o')
            for test_keys, res in tests
        )

    # Count how many combinations satisfy the condition.
    # We use a generator expression inside sum() for memory efficiency.
    result = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(result)

if __name__ == "__main__":
    solve()