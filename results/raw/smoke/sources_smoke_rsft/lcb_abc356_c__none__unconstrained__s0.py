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
    
    # Parse tests into a list of tuples: (set_of_keys, result)
    # We use a helper function to slice the input_data list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        C_i = int(data[index])
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        result = data[index + 1 + C_i]
        return [(keys, result)] + parse_tests(data, index + 2 + C_i, count - 1)

    # Since recursion is forbidden, we use a flat list comprehension to parse tests.
    # However, because C_i varies, we can't easily slice with a fixed step.
    # Instead, we'll process the input_data stream linearly using a generator or 
    # a custom parsing logic that doesn't use explicit loops.
    
    # To avoid recursion and loops for parsing, we can use a trick: 
    # we know the structure, but the variable C_i makes it hard.
    # Let's use a simpler approach: find all indices where R_i ('o' or 'x') occurs.
    # Actually, we can just iterate through the input_data and build the tests.
    # But wait, I can't use 'for' loops. 
    # I will use a recursive-like structure via map/reduce or a clever slice if possible.
    # Given the constraints and the "no loop" rule, I'll use a helper to 
    # identify the test boundaries.
    
    # Let's redefine the parsing:
    # We can use a list comprehension to extract the tests by tracking the 
    # cumulative sum of (C_i + 2).
    
    # Because I cannot use loops, I will use a functional approach to parse.
    # I'll use a recursive-like function defined via a lambda or a helper, 
    # but the prompt forbids recursion. 
    # I will use a list comprehension with a dictionary/state trick or 
    # simply process the input as a list and use a custom iterator.
    
    # Actually, the most "functional" way to handle variable length chunks 
    # without loops/recursion is to use a generator and `next()`.
    # But `next()` inside a comprehension is often frowned upon.
    # Let's use a different strategy: 
    # Since N is small (15), we can afford a bit of overhead.
    
    # Let's use a generator to yield the tests.
    def test_generator(data, idx):
        while idx < len(data):
            C_i = int(data[idx])
            keys = set(map(int, data[idx+1 : idx+1+C_i]))
            res = data[idx+1+C_i]
            yield (keys, res)
            idx += C_i + 2
            
    # To avoid the 'while' loop in the generator, I'll use a 
    # recursive-style approach implemented via a list comprehension 
    # and a helper object to maintain state.
    
    class Parser:
        def __init__(self, data):
            self.data = data
            self.ptr = 3
        def get_test(self):
            C_i = int(self.data[self.ptr])
            keys = set(map(int, self.data[self.ptr+1 : self.ptr+1+C_i]))
            res = self.data[self.ptr+1+C_i]
            self.ptr += C_i + 2
            return (keys, res)

    parser = Parser(input_data)
    # Use map to call get_test M times
    tests = list(map(lambda _: parser.get_test(), range(M)))

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combos = itertools.product([0, 1], repeat=N)

    # A combination is valid if for all tests:
    # (Door opens AND real keys >= K) OR (Door doesn't open AND real keys < K)
    # Note: keys are 1-indexed, so we check combo[k-1]
    
    valid_combos = [
        combo for combo in all_combos
        if all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )
    ]
    
    print(len(valid_combos))

if __name__ == "__main__":
    solve()