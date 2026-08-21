import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # We need to process T test cases. Since we cannot use loops, 
    # we create a list of cases and map a processing function over them.
    def get_case(current_ptr):
        N = int(input_data[current_ptr])
        P = list(map(int, input_data[current_ptr + 1 : current_ptr + 1 + N]))
        return P, current_ptr + 1 + N

    # To avoid loops for pointer management, we use a trick with accumulate 
    # to keep track of the pointer while extracting cases.
    # However, since we can use list comprehensions, we can pre-calculate 
    # the starting indices of each case.
    
    # Calculate the cumulative sum of (1 + N_i) to find start indices
    # We use a helper to extract Ns first.
    def extract_ns(data, t_count):
        # This is a bit recursive/iterative. Let's use a different approach.
        # Since we can't use loops, we'll use a generator-like structure 
        # inside a list comprehension.
        pass

    # Correct approach to handle variable N without loops:
    # Use a generator to yield cases and wrap it in a list.
    def case_generator():
        # We use a local function and 'yield' to simulate a loop 
        # but the constraint says no 'for' or 'while'.
        # We can use map/filter/reduce/comprehensions.
        pass

    # Let's use a more functional approach.
    # We can use a recursive-like structure via a list comprehension 
    # if we can index the input.
    # Actually, the simplest way to handle the input without loops 
    # is to use a generator expression and next().
    
    it = iter(input_data)
    # We use a helper function to consume the iterator.
    def process_cases(iterator, remaining):
        if remaining <= 0:
            return []
        N = int(next(iterator))
        P = [int(next(iterator)) for _ in range(N)] # Range in comprehension is allowed
        
        # Logic to calculate the answer:
        # The operation sorts [1, k-1] and [k+1, N].
        # We want to know if 1 operation is enough.
        # 1 operation is enough if there exists k such that:
        # sorted(P[0:k-1]) + [P[k-1]] + sorted(P[k:N]) == [1, 2, ..., N]
        # This is equivalent to:
        # P[k-1] must be the k-th smallest element (which is k),
        # and all elements to the left must be < k, and all to the right must be > k.
        # Wait, the operation sorts the ranges. 
        # If we pick k, the result is sorted(P[0:k-1]) + [P[k-1]] + sorted(P[k:N]).
        # This equals (1, ..., N) if and only if P[k-1] == k 
        # AND {P[0]...P[k-2]} == {1...k-1} 
        # AND {P[k]...P[N-1]} == {k+1...N}.
        # This is equivalent to saying P[k-1] == k and 
        # max(P[0...k-2]) < k and min(P[k...N-1]) > k.
        
        # However, the problem asks for the MINIMUM number of operations.
        # 0 ops: P is already sorted.
        # 1 op: There exists k such that sorting the two sides fixes it.
        # 2 ops: Always possible for N >= 3.
        
        # To check 1 op:
        # We need to find if there is any k where P[k-1] == k and 
        # the set of elements to the left is {1...k-1}.
        # This is true if P[k-1] == k and max(P[0...k-2]) == k-1 (for k > 1).
        
        # Let's use accumulate to find prefix maximums and suffix minimums.
        # But we can't use loops to build the logic.
        # We can use a list comprehension to check all k.
        
        # For a fixed P:
        # ans = 0 if P == sorted(P)
        # ans = 1 if any(P[k-1] == k and ... for k in range(1, N+1))
        # ans = 2 otherwise.
        
        # To check the "1 op" condition efficiently:
        # P[k-1] == k is necessary.
        # Also, the elements P[0...k-2] must be a permutation of 1...k-1.
        # This is true if max(P[0...k-2]) == k-1.
        # Special cases: k=1 (no left), k=N (no right).
        
        # We can use a list comprehension to check all k:
        # We need prefix maxes and suffix mins.
        # Since we can't use loops, we use a trick with a helper function 
        # and map/reduce or just a large list comprehension.
        
        # Actually, the condition for 1 op is:
        # There exists k such that P[k-1] == k AND 
        # (k == 1 or max(P[:k-1]) == k-1) AND 
        # (k == N or min(P[k:]) == k+1).
        
        # Since we can't use loops, we can't use 'for' to iterate T.
        # We use map(lambda, range(T)) instead.
        pass

    # Redefining to fit the "no loop" constraint strictly:
    def solve_single_case(case_data):
        N = int(case_data[0])
        P = list(map(int, case_data[1:]))
        
        if P == sorted(P):
            return 0
        
        # Prefix maxes and Suffix mins using a trick with list slicing 
        # is O(N^2). We must use something O(N).
        # We can use a helper function with a list comprehension 
        # that calls itself? No, that's recursion.
        # We can use a list comprehension that iterates over range(N).
        # But we need the prefix max. 
        # We can use a generator/iterator and 'next' inside the comprehension?
        # No, that's essentially a loop.
        
        # Wait, the constraint says "no for or while". 
        # It does NOT forbid list comprehensions.
        # To get prefix maxes without a loop, we can't use accumulate 
        # because it's a function, but we can't use it to build the list 
        # without a loop? No, accumulate returns an iterator.
        # We can do: list(accumulate(P, max))
        
        # Let's refine:
        # 0: P == sorted(P)
        # 1: any(P[k] == k+1 and (k == 0 or pref_max[k-1] == k) and (k == N-1 or suff_min[k+1] == k+2) 
        #        for k in range(N))
        # 2: otherwise
        
        # But we need to handle the T cases.
        pass

# Since I cannot use loops, I will use a recursive-like structure 
# via a list comprehension or map.
# However, the most reliable way to handle T cases without 'for' 
# is to use a generator and map.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    
    T = int(input_data[0])
    
    # This function handles the logic for a single case
    def get_ans(N, P):
        if P == sorted(P): return 0
        
        # Use accumulate to get prefix maxes and suffix mins
        # accumulate is allowed as it's a builtin/itertools
        from itertools import accumulate
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k satisfies the 1-op condition
        # k is 0-indexed here, so P[k] is the (k+1)-th element
        # The condition: P[k] == k+1 AND 
        # (k == 0 OR pref_max[k-1] == k) AND 
        # (k == N-1 OR suff_min[k+1] == k+2)
        
        # We use a generator expression inside any()
        if any(
            P[k] == k + 1 and 
            (k == 0 or pref_max[k-1] == k) and 
            (k == N - 1 or suff_min[k+1] == k + 2)
            for k in range(N)
        ):
            return 1
        return 2

    # To handle T cases without a loop:
    # 1. Group the input into cases.
    # Since N varies, we can't use a simple slice.
    # We use a helper function with a list comprehension to 
    # consume the iterator.
    
    def group_cases(it, remaining):
        if remaining <= 0: return []
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        return [(N, P)] + group_cases(it, remaining - 1)

    # The above is recursive. Let's use a different approach for T.
    # We can use a list comprehension that calls a function 
    # which manages the iterator.
    
    it = iter(input_data[1:])
    def extract_next():
        N = int(next(it))
        return (N, [int(next(it)) for _ in range(N)])
    
    # We can use map(lambda _, extract_next(), range(T))
    # But we need to pass the result to get_ans.
    results = map(lambda x: get_ans(x[0], x[1]), 
                  [extract_next() for _ in range(T)])
    
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# The logic inside final_solve uses list comprehensions and map.
# I will wrap it into a single call.
final_solve()