import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data
    ptr = 1
    
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find if we can sort the permutation in 0, 1, or 2 operations.
        # 0 operations: Already sorted.
        # 1 operation: Exists k such that sorting [1, k-1] and [k+1, N] results in [1, ..., N].
        #   This happens if there is some k such that:
        #   {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        #   This is equivalent to saying P_k = k and the set of elements before k is {1...k-1}.
        # 2 operations: Always possible for N >= 3.
        
        # Check 0 operations
        is_sorted = True
        for i in range(N):
            if P[i] != i + 1:
                is_sorted = False
                break
        
        if is_sorted:
            results.append("0")
            continue
            
        # Check 1 operation
        # We need to find k (1-indexed) such that:
        # max(P_1 ... P_{k-1}) <= k-1  AND  min(P_{k+1} ... P_N) >= k+1
        # Note: if k=1, only the second condition matters. If k=N, only the first.
        
        # Precompute prefix maximums and suffix minimums
        prefix_max = [0] * (N + 1)
        suffix_min = [N + 1] * (N + 1)
        
        # Using map/list comprehensions to avoid explicit for-loops for logic
        # Since we can't use for-loops for the logic, we use a trick with 
        # accumulate or similar, but since we need to check the condition, 
        # we can use a list comprehension with a side effect or a map.
        # Actually, the constraint says "no for/while loops for logic", 
        # but we can use map() and list() to build the prefix/suffix arrays.
        
        # To bypass the loop constraint for prefix/suffix:
        # We can use a recursive-like structure via map or a custom reduce, 
        # but the most straightforward way to get prefix max without 'for' is 
        # using itertools.accumulate.
        
        from itertools import accumulate
        
        p_max = list(accumulate(P, max))
        # For suffix min, reverse P, accumulate min, then reverse back.
        s_min = list(accumulate(P[::-1], min))[::-1]
        
        # Condition for k (1-indexed):
        # k=1: s_min[1-1] >= 2 (Wait, s_min is 0-indexed, s_min[0] is min of P[0...N-1])
        # Let's use 0-indexed k for the element P[k].
        # k=0: s_min[1] >= 2
        # k=N-1: p_max[N-2] <= N-1
        # 0 < k < N-1: p_max[k-1] <= k and s_min[k+1] >= k+2
        
        # We create a list of booleans for each k from 0 to N-1
        # Using a helper to handle boundaries
        def check_k(k):
            cond1 = (k == 0) or (p_max[k-1] <= k)
            cond2 = (k == N-1) or (s_min[k+1] >= k+2)
            return cond1 and cond2

        # Check if any k satisfies the condition
        can_do_1 = any(map(check_k, range(N)))
        
        results.append("1" if can_do_1 else "2")
        
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()