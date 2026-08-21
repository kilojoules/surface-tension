import sys

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
    # Each test is structured as: C_i, then C_i keys, then R_i
    # We need to group these since C_i varies.
    # We use a helper function or a comprehension to slice the input_data.
    
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = [int(x) for x in data[index + 1 : index + 1 + c_i]]
        # Result is at index + 1 + c_i
        res = data[index + 1 + c_i]
        
        # Recurse is forbidden, so we must handle this differently.
        # Since M is small (100), we can pre-calculate the starting indices of each test.
        return (keys, res)

    # To avoid recursion and loops, we calculate the starting position of each test
    # using a cumulative sum logic via a list comprehension.
    # However, since we can't use loops to find indices, we can use a 
    # trick with a list to store the tests.
    
    # Let's redefine how we extract tests. 
    # We can use a generator-like approach within a list comprehension.
    # But the simplest way to avoid loops/recursion for parsing variable length 
    # input is to use a custom iterator.
    it = iter(input_data[3:])
    
    # We can't use a for loop to call next(it), but we can use 
    # a list comprehension that calls a function.
    def get_test(iterator):
        c_i = int(next(iterator))
        keys = [int(next(iterator)) for _ in range(c_i)]
        res = next(iterator)
        return (keys, res)

    # Since we can't use 'for' or 'while', we use map() or list comprehension.
    # To call get_test M times, we can use range(M).
    tests = [get_test(it) for _ in range(M)]

    # There are 2^N possible combinations of real/dummy keys.
    # We represent a combination as a bitmask from 0 to 2^N - 1.
    # Bit j is 1 if key (j+1) is real, 0 otherwise.
    
    # Check if a specific combination (mask) satisfies a specific test
    def check(mask, test):
        keys, res = test
        # Count how many keys in the test are real in this mask
        # Key k is real if (mask >> (k-1)) & 1 == 1
        real_count = sum([1 for k in keys if (mask >> (k-1)) & 1])
        
        if res == 'o':
            return real_count >= K
        else:
            return real_count < K

    # Count combinations that satisfy all tests
    # We use a list comprehension to iterate 2^N and sum the booleans
    ans = sum([
        1 for mask in range(1 << N)
        if all(check(mask, t) for t in tests)
    ])
    
    print(ans)

if __name__ == "__main__":
    solve()