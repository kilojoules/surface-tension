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
    # We use a helper function to slice the input list based on C_i
    def parse_tests(data, index, count):
        if count == 0:
            return []
        
        # C_i is at data[index]
        c_i = int(data[index])
        # Keys are from index + 1 to index + c_i
        keys = set(map(int, data[index + 1 : index + 1 + c_i]))
        # Result is at index + 1 + c_i
        res = data[index + 1 + c_i]
        
        return [(keys, res)] + parse_tests(data, index + 2 + c_i, count - 1)

    # Since we cannot use loops, we use a recursive-like structure or 
    # a comprehension to parse the tests. 
    # However, the simplest way to handle the variable length of C_i 
    # without loops is to pre-process the input into a structured format.
    
    # Re-evaluating parsing: since we can't use while/for, 
    # we can use a generator or a complex comprehension.
    # Let's use a more robust approach to extract tests.
    
    def get_tests():
        # We use a generator to yield tests, then convert to list
        # But wait, the prompt says no for/while. 
        # We can use map/filter/reduce or recursion.
        def recurse(idx, remaining):
            if remaining <= 0:
                return []
            c_i = int(input_data[idx])
            keys = set(map(int, input_data[idx+1 : idx+1+c_i]))
            res = input_data[idx+1+c_i]
            return [(keys, res)] + recurse(idx + 2 + c_i, remaining - 1)
        return recurse(3, M)

    tests = get_tests()

    # Generate all 2^N combinations of real (1) and dummy (0) keys
    # Each combination is a tuple of length N
    all_combinations = itertools.product([0, 1], repeat=N)

    # Validation function for a single combination
    # combo is a tuple where index i corresponds to key i+1
    def is_valid(combo):
        # Check if all tests are satisfied
        # For each test:
        # If R_i == 'o', sum of real keys in set must be >= K
        # If R_i == 'x', sum of real keys in set must be < K
        return all(
            (sum(combo[k-1] for k in test_keys) >= K) if res == 'o' 
            else (sum(combo[k-1] for k in test_keys) < K)
            for test_keys, res in tests
        )

    # Count valid combinations using map and sum
    # We use a generator expression inside sum()
    result = sum(1 for combo in all_combinations if is_valid(combo))
    
    print(result)

if __name__ == "__main__":
    solve()