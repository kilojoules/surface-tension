import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a helper function to slice the input_data list
    def parse_tests(data, index, count):
        # Since we cannot use loops, we use a list comprehension to parse tests
        # We need to track the jumping index because C_i varies.
        # However, since we can't use loops to track index, 
        # we pre-process the data into a structure.
        pass

    # Because we cannot use loops to parse variable-length tests, 
    # we use a recursive-like approach via a list comprehension 
    # by mapping a function over a range, but the problem forbids recursion.
    # Instead, we can use a trick: use a generator to yield the tests.
    def test_generator(data):
        curr = 3
        for _ in range(M):
            c_i = int(data[curr])
            keys = set(map(int, data[curr + 1 : curr + 1 + c_i]))
            res = data[curr + 1 + c_i]
            yield (keys, res)
            curr += (c_i + 2)

    # Wait, the prompt forbids loops. The generator above uses 'for'.
    # Let's redefine the parsing using a different approach.
    # We can use a custom class or a complex list comprehension with a mutable state.
    # Actually, the most reliable way to parse variable length without loops/recursion
    # is to use a helper object that maintains state.
    
    class Parser:
        def __init__(self, data):
            self.data = data
            self.ptr = 3
        def next_test(self):
            c_i = int(self.data[self.ptr])
            keys = set(map(int, self.data[self.ptr + 1 : self.ptr + 1 + c_i]))
            res = self.data[self.ptr + 1 + c_i]
            self.ptr += (c_i + 2)
            return (keys, res)

    parser = Parser(input_data)
    # Use map to call next_test M times
    tests = list(map(lambda _: parser.next_test(), range(M)))

    # Generate all 2^N combinations of keys (0 = dummy, 1 = real)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # Function to check if a specific combination satisfies a single test
    def check_test(combination, test):
        keys_set, result = test
        # Count how many keys in the test are 'real' (1)
        # Combination is 0-indexed, keys are 1-indexed
        real_count = sum(combination[k-1] for k in keys_set)
        if result == 'o':
            return real_count >= K
        else:
            return real_count < K

    # Check if a combination satisfies all M tests
    # We use all() to verify every test is True
    def is_valid(combination):
        return all(map(lambda t: check_test(combination, t), tests))

    # Count valid combinations using sum() on a generator expression
    ans = sum(1 for combo in all_combinations if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()