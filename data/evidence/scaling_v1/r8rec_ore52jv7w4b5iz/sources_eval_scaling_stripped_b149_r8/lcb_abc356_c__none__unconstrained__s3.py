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
    # we need a way to parse the tests.
    # We use a generator to yield the tests one by one.
    def get_tests(data):
        it = iter(data[3:])
        for _ in range(M):
            try:
                c_i = int(next(it))
                keys = [int(next(it)) for _ in range(c_i)]
                result = next(it)
                yield (keys, result)
            except StopIteration:
                break

    tests = list(get_tests(input_data))
    
    # Generate all 2^N possible combinations of keys (0: dummy, 1: real)
    # product([0, 1], repeat=N) generates tuples of length N
    all_combinations = product([0, 1], repeat=N)
    
    # For a combination to be valid, it must satisfy all M tests.
    # A test is satisfied if:
    # (result == 'o' and count of real keys in set >= K) OR
    # (result == 'x' and count of real keys in set < K)
    
    # We use a generator expression inside sum() to count valid combinations.
    # combination is a tuple where index i corresponds to key i+1.
    # We use map(lambda k: combination[k-1], test_keys) to get the status of keys in the test.
    
    ans = sum(
        1 for combination in all_combinations
        if all(
            (sum(combination[k-1] for k in test_keys) >= K) if result == 'o' 
            else (sum(combination[k-1] for k in test_keys) < K)
            for test_keys, result in tests
        )
    )
    
    print(ans)

if __name__ == "__main__":
    solve()