import sys
from itertools import product

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Since the input format for tests is variable length (C_i), 
    # we need a way to group the tests.
    # We can use a generator or a loop to extract tests from the flat list.
    def get_tests(data):
        it = iter(data[3:])
        for _ in range(M):
            try:
                c = int(next(it))
                keys = [int(next(it)) for _ in range(c)]
                result = next(it)
                yield (keys, result)
            except StopIteration:
                break

    tests = list(get_tests(input_data))

    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for every test:
    # If result is 'o', sum of real keys in the test set >= K
    # If result is 'x', sum of real keys in the test set < K
    # Note: keys are 1-indexed, so we use key-1 for 0-indexing
    
    def is_valid(combo):
        for keys, result in tests:
            # Count how many keys in this test are real in the current combination
            real_count = sum(combo[k-1] for k in keys)
            if result == 'o':
                if real_count < K:
                    return False
            else: # result == 'x'
                if real_count >= K:
                    return False
        return True

    # Count valid combinations using a generator expression inside sum()
    ans = sum(1 for combo in all_combinations if is_valid(combo))
    print(ans)

if __name__ == "__main__":
    solve()