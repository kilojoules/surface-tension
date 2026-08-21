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

    # Parse the tests into a list of tuples: (set_of_keys, result)
    # We use a helper function or a complex comprehension to group the variable-length input
    # Since we cannot use loops, we process the input list by tracking indices.
    # However, a cleaner way to handle the variable length is to pre-process the 
    # input into a structured format using a custom generator or mapping.
    
    # To avoid loops for parsing, we can use a recursive-like structure via 
    # a list comprehension that consumes the input list. 
    # But since we can't use recursion, we use a trick with a generator 
    # and next() inside a list comprehension.
    
    it = iter(input_data[3:])
    
    # This list comprehension effectively parses the M tests
    # For each test: read C_i, then read C_i keys, then read R_i
    # We use a helper list to maintain state across the comprehension
    def parse_tests(iterator, count):
        # We use a list comprehension to drive the parsing
        # Each element is a tuple (keys_set, result)
        # We use a list to simulate a mutable state for the iterator
        return [
            (
                {int(next(iterator)) for _ in range(int(next(it_copy)))} 
                if (it_copy := iterator) else set(), 
                next(iterator)
            ) 
            for _ in range(count)
        ]
    
    # The above logic is tricky without loops. Let's use a different approach:
    # Since M is small (100), we can use a list comprehension that 
    # calculates the starting index of each test block.
    # But the blocks are variable length. 
    # Let's use a more robust way to parse:
    
    # We can use a generator function to yield the tests, 
    # and then wrap it in a list. Generator functions are allowed 
    # as long as they don't use 'for' or 'while' loops.
    # Wait, the prompt says no loops. Generators usually use loops.
    # Let's use a recursive-like structure via map/reduce or 
    # simply process the flat list by calculating offsets.
    
    # Actually, the simplest way to parse variable length input without 
    # for/while is to use a recursive function, but recursion is banned.
    # Let's use a trick: use a list comprehension and a mutable object (list) 
    # to keep track of the current index.
    
    idx = [0]
    def get_test():
        # Read C_i
        c = int(input_data[3 + idx[0]])
        # Extract keys
        keys = {int(x) for x in input_data[4 + idx[0] : 4 + idx[0] + c]}
        # Extract result
        res = input_data[4 + idx[0] + c]
        # Update index: C_i + 2 (for C_i and R_i)
        idx[0] += c + 2
        return (keys, res)

    # We can use map() to call get_test M times
    tests = list(map(lambda _: get_test(), range(M)))

    # Iterate through all 2^N combinations
    # A combination is represented by an integer i from 0 to 2^N - 1
    # Bit j of i is 1 if key (j+1) is real, 0 otherwise.
    
    # Check if a combination i satisfies test (keys, res)
    # Number of real keys in the test = count of bits set in (i & mask)
    # where mask has bits set for the keys in the test.
    
    # Pre-calculate masks for each test
    test_masks = [
        (sum(1 << (k-1) for k in keys), res)
        for keys, res in tests
    ]
    
    # A combination i is valid if for all (mask, res):
    # if res == 'o', bin(i & mask).count('1') >= K
    # if res == 'x', bin(i & mask).count('1') < K
    
    # Use a list comprehension to check all 2^N combinations
    # and sum the booleans (True=1, False=0)
    ans = sum(
        all(
            (bin(i & mask).count('1') >= K) if res == 'o' else (bin(i & mask).count('1') < K)
            for mask, res in test_masks
        )
        for i in range(1 << N)
    )
    
    print(ans)

if __name__ == "__main__":
    solve()