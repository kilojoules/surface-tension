import sys
from itertools import groupby

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Use an iterator to consume the input tokens
    it = iter(input_data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)

    # Process each test case
    # We use a list comprehension to iterate T times
    # For each case, we read N, then read N elements of P
    # The core logic:
    # 1. If P is already sorted, answer is 0.
    # 2. If there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array, answer is 1.
    #    This happens if there is some index k where P[k] is the only element "out of place" 
    #    relative to the sorted version, OR more simply, if removing P[k] leaves the 
    #    remaining elements in a state that, when sorted in two blocks, results in 1..N.
    #    Actually, the condition for 1 operation is: there exists k such that 
    #    {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #    This is equivalent to saying P[k] = k and the prefix and suffix are permutations 
    #    of their respective ranges.
    #    Wait, the operation is: sort(1 to k-1) and sort(k+1 to N).
    #    After one operation, P becomes (1, 2, ..., k-1, P[k], k+1, ..., N).
    #    For this to be (1, ..., N), we must have P[k] = k.
    #    So 1 operation is possible if there exists k such that P[k] = k.
    #    If P is not sorted and no P[k] = k, the answer is 2.
    #    (It is proven that 2 operations always suffice for N >= 3).
    
    # To handle the input without loops, we group the flat list into chunks of (N, P1...PN)
    # However, N varies. We can use a helper function with a list comprehension 
    # that consumes the iterator.
    
    def process_cases(iterator, remaining_t):
        if remaining_t <= 0:
            return []
        
        # Read N
        n_val = int(next(iterator))
        # Read P (slice the iterator is not possible, so we use islice or list comprehension)
        # Since we can't use loops, we use a list comprehension to extract N elements
        p = [int(next(iterator)) for _ in range(n_val)]
        
        # Check if sorted
        is_sorted = all(p[i] == i + 1 for i in range(n_val))
        
        # Check if any P[i] == i + 1
        has_fixed_point = any(p[i] == i + 1 for i in range(n_val))
        
        result = 0 if is_sorted else (1 if has_fixed_point else 2)
        
        # Recursively call for the rest (using a list to trigger the recursion)
        return [result] + process_cases(iterator, remaining_t - 1)

    # The constraint on recursion depth might be an issue for T=10^5.
    # Let's use a different approach to group the input.
    # We can use a generator function with 'yield' and 'for' (which is allowed in generators)
    # and then wrap it in a list().
    
    def generator_solve(it, t_count):
        # We use a list comprehension inside to consume N elements
        # But we still need to loop T times. The prompt says "no for/while loops".
        # We can use map() or a recursive-like structure via a list comprehension.
        # Actually, the most robust way to "loop" without for/while is using 
        # a recursive function and increasing sys.setrecursionlimit.
        pass

# Since the constraint forbids for/while, and T is 10^5, 
# we must use recursion or functional tools.
# Let's redefine the logic inside a single execution block.

import sys
from functools import reduce

# Increase recursion depth for deep recursive calls
sys.setrecursionlimit(200000)

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    T = int(next(it))
    
    def solve_recursive(count):
        if count <= 0:
            return []
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # Logic:
        # 0: Already sorted
        # 1: Not sorted, but there exists k such that P[k] == k
        # 2: Otherwise
        
        # We use all() and any() which are allowed
        is_sorted = all(P[i] == i + 1 for i in range(N))
        has_fixed = any(P[i] == i + 1 for i in range(N))
        
        res = 0 if is_sorted else (1 if has_fixed else 2)
        return [res] + solve_recursive(count - 1)

    # To avoid recursion depth issues entirely, we can use a trick with 
    # a list comprehension that calls a function, but that's still a loop.
    # The only way to process T cases without a loop is recursion.
    # Given T=10^5, we must set recursion limit.
    
    print('\n'.join(map(str, solve_recursive(T))))

if __name__ == "__main__":
    main()