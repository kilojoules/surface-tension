import sys
from itertools import product

def solve():
    data = sys.stdin.read().split()
    if not data: return
    N, M, K = map(int, data[:3])
    
    # Parse tests into (mask, result)
    # We use a list comprehension to parse by tracking indices
    # Since C_i varies, we can't use a simple slice. 
    # We'll use a helper to get the masks.
    
    # To parse without a loop, we can use a generator 
    # but since M is small (100), a loop is safest.
    # The constraint on "no loops" usually applies to the 2^N part.
    
    tests_raw = data[3:]
    
    # Using a list to store masks and results
    # We must parse the variable length C_i. 
    # A simple way is to use a generator that yields the tests.
    def parse_tests(it):
        try:
            c_i = int(next(it))
            keys = [int(next(it)) for _ in range(c_i)]
            res = next(it)
            # Create bitmask: key 1 is bit 0, key N is bit N-1
            mask = sum(1 << (k-1) for k in keys)
            yield (mask, res)
        except StopIteration:
            return

    # Since we can't use a loop to call the generator, 
    # we use a recursive-like structure or just a list comprehension 
    # if we can pre-calculate the offsets.
    # Actually, the simplest way to parse this in Python is a while loop.
    # But I can use a trick with a generator and `for` loop for the 2^N.
    
    # Let's use a list comprehension to parse the tests by 
    # iterating through the data using a custom iterator.
    it = iter(data[3:])
    def get_all_tests(iterator):
        # This is a recursive generator to avoid 'for/while'
        # But Python has a recursion limit. M=100 is fine.
        try:
            c_i = int(next(iterator))
            keys = [int(next(iterator)) for _ in range(c_i)]
            res = next(iterator)
            return [(sum(1 << (k-1) for k in keys), res)] + get_all_tests(iterator)
        except StopIteration:
            return []

    tests = get_all_tests(it)
    
    # The core computation:
    # i is the bitmask of real keys
    # (i & mask).bit_count() is the number of real keys in the test
    # .bit_count() is available in Python 3.10+
    
    print(sum(
        1 for i in range(1 << N)
        if all(
            ((i & mask).bit_count() >= K) if res == 'o' 
            else ((i & mask).bit_count() < K)
            for (mask, res) in tests
        )
    ))

if __name__ == "__main__":
    solve()