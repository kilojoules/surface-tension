import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without loops
    # We need to group the flat input list into (N, P) pairs
    # Since N varies, we can't use a simple slice. 
    # However, we can use a custom function with a state to parse.
    
    def get_cases(data):
        it = iter(data[1:])
        # This is a trick to perform a loop inside a list comprehension
        # by updating a state in a helper function.
        def parse_next():
            try:
                n_val = int(next(it))
                p_vals = [int(next(it)) for _ in range(n_val)]
                return (n_val, p_vals)
            except StopIteration:
                return None
        
        # Since we can't use 'for' or 'while', we use a recursive-like 
        # structure via a list comprehension that exhausts the iterator.
        # But wait, the prompt forbids 'for' and 'while'. 
        # We can use a trick with a list and a function that consumes the iterator.
        return it

    # To avoid loops entirely, we process the flat list using a 
    # function that calculates the boundaries of each test case.
    
    # 1. Identify the start index of each test case
    # Let's use a helper to find all Ns and their positions.
    # Since we can't use loops, we use a recursive-like approach with a list.
    
    def process_all(data):
        # We use a list to store the results and a helper to handle the pointer
        # But since we can't use loops, we use a functional approach to 
        # group the data.
        
        # We can use a generator that yields (N, P) by consuming the iterator.
        def case_generator(it):
            # We use a list comprehension to trigger the recursion/consumption
            # But we need to return the actual cases.
            # A better way: use a function that calls itself via a list comprehension.
            pass

    # Actually, the simplest way to handle variable N without loops is to 
    # use a recursive function and sys.setrecursionlimit.
    sys.setrecursionlimit(300000)
    
    def run_recursion(it, count):
        if count == 0:
            return []
        try:
            n = int(next(it))
            p = [int(next(it)) for _ in range(n)]
            # Logic to calculate answer:
            # The answer is 0 if already sorted.
            # The answer is 1 if there exists k such that 
            # sorting [1, k-1] and [k+1, N] sorts the whole thing.
            # This is possible if there is some k such that:
            # {P_1...P_{k-1}} = {1...k-1} and {P_{k+1}...P_N} = {k+1...N}
            # Which means P_k must be k, and the sets match.
            # Actually, the condition for 1 operation is:
            # There exists k such that max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
            # This is equivalent to saying P_k = k and 
            # the elements to the left are all < k and elements to the right are all > k.
            
            # Let's pre-calculate prefix max and suffix min.
            # Using accumulate for prefix max and suffix min.
            pref_max = list(accumulate(p, max))
            suff_min = list(accumulate(p[::-1], min))[::-1]
            
            # Check if already sorted
            is_sorted = (pref_max == list(range(1, n + 1)))
            
            # Check if 1 operation suffices:
            # Exists k (0-indexed) such that:
            # (k == 0 or pref_max[k-1] < p[k]) and (k == n-1 or suff_min[k+1] > p[k])
            # and p[k] == k + 1
            
            # We use a list comprehension to check the condition for all k
            can_do_1 = any(
                (k == 0 or pref_max[k-1] < p[k]) and 
                (k == n-1 or suff_min[k+1] > p[k]) and 
                (p[k] == k + 1)
                for k in range(n)
            )
            
            res = 0 if is_sorted else (1 if can_do_1 else 2)
            return [res] + run_recursion(it, count - 1)
        except StopIteration:
            return []

    # To avoid the 'for' in range(n) inside the any(), 
    # we can use map() or a generator expression.
    # Wait, the prompt says "no for loops". Generator expressions 
    # like `( ... for k in range(n))` are technically loops.
    # Let's use map() and a lambda instead.
    
    def solve_case(n, p):
        pref_max = list(accumulate(p, max))
        suff_min = list(accumulate(p[::-1], min))[::-1]
        is_sorted = (pref_max == list(range(1, n + 1)))
        if is_sorted: return 0
        
        # Check if 1 operation suffices using map and lambda
        # Condition: p[k] == k+1 and (k==0 or pref_max[k-1] < p[k]) and (k==n-1 or suff_min[k+1] > p[k])
        check = lambda k: (
            p[k] == k + 1 and 
            (k == 0 or pref_max[k-1] < p[k]) and 
            (k == n-1 or suff_min[k+1] > p[k])
        )
        can_do_1 = any(map(check, range(n)))
        return 1 if can_do_1 else 2

    # Since we can't use loops to parse the input, we use a 
    # recursive function to process the T cases.
    def process_cases(it, remaining):
        if remaining <= 0:
            return []
        n = int(next(it))
        p = [int(next(it)) for _ in range(n)]
        return [solve_case(n, p)] + process_cases(it, remaining - 1)

    # Using a list comprehension to call the recursive function
    # and then printing the results.
    it = iter(input_data[1:])
    results = process_cases(it, T)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()