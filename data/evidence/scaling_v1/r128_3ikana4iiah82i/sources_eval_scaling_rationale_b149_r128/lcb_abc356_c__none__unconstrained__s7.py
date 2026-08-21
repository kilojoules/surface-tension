import sys
from itertools import product

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, M, and K
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])

    # Parse the tests
    # Each test is stored as (set_of_keys, result)
    # We use a generator to group the flat input list into test structures
    def parse_tests(data, index):
        if index >= len(data):
            return []
        
        # C_i is the number of keys in the test
        C_i = int(data[index])
        # The keys are the next C_i elements
        keys = set(map(int, data[index + 1 : index + 1 + C_i]))
        # The result is the element after the keys
        result = data[index + 1 + C_i]
        
        return [(keys, result)] + parse_tests(data, index + 2 + C_i)

    # Since we cannot use loops, we use a list comprehension to organize the tests
    # However, the recursive parse_tests is cleaner for variable-length inputs.
    # Let's redefine test parsing using a more robust method.
    
    # To avoid recursion limits and loops, we can use a trick with a helper 
    # function and a list comprehension if we know the structure.
    # But since C_i varies, we must process the list linearly.
    # Let's use a while-loop replacement via a custom iterator.
    
    it = iter(input_data[3:])
    def get_tests():
        # We use a list comprehension with a helper that consumes the iterator
        # Note: calling next() inside a list comprehension is a common way 
        # to simulate a loop while staying within a single expression.
        return [
            (set(map(int, [next(it) for _ in range(int(next(it)))])) if False else 
             # The above line is a dummy to trigger the range. 
             # Actually, the most reliable way to parse this without 'for' 
             # loops is to use a recursive-like structure or a map.
             None) 
            for _ in range(M)
        ]
    
    # Correcting the parse logic: since we can't use 'for' loops, 
    # we use a recursive function to build the test list.
    def build_tests(tokens):
        if not tokens:
            return []
        c = int(tokens[0])
        keys = set(tokens[1 : 1 + c])
        res = tokens[1 + c]
        return [(keys, res)] + build_tests(tokens[2 + c :])

    tests = build_tests(input_data[3:])

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = product([0, 1], repeat=N)

    # A combination is valid if for all tests:
    # (sum of real keys in test >= K) == (result == 'o')
    # We use a generator expression inside sum() to count real keys.
    # The keys in 'tests' are strings from the input, so we map them to indices.
    
    # We use a list comprehension to count how many combinations satisfy all tests.
    # combo is a tuple where combo[i] is the status of key i+1.
    ans = sum(
        1 for combo in all_combinations
        if all(
            (sum(combo[int(k) - 1] for k in test_keys) >= K) == (res == 'o')
            for test_keys, res in tests
        )
    )

    print(ans)

if __name__ == "__main__":
    solve()