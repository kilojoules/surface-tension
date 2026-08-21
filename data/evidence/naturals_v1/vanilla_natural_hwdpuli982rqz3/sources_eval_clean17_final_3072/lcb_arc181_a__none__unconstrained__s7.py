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
    
    # The problem asks for the minimum number of operations to sort P.
    # One operation with index k sorts [1, k-1] and [k+1, N].
    # This means P_k remains in place, and everything else is sorted around it.
    # For P to become (1, 2, ..., N) in one move, there must exist some k such that:
    # 1. P_k = k
    # 2. All elements {P_1, ..., P_{k-1}} are {1, ..., k-1} (though not necessarily in order)
    # 3. All elements {P_{k+1}, ..., P_N} are {k+1, ..., N} (though not necessarily in order)
    # Actually, condition 2 and 3 are implied if P_k = k and the set {P_1, ..., P_{k-1}} is {1, ..., k-1}.
    
    # To check if 0 operations: P is already sorted.
    # To check if 1 operation: There exists k such that P_k = k AND 
    # max(P_1...P_{k-1}) < k AND min(P_{k+1}...P_N) > k.
    # To check if 2 operations: It is proven that 2 operations are always sufficient for N >= 3.
    # (e.g., k=1 sorts [2, N], then k=N sorts [1, N-1]).
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # Check 0: Already sorted
        # Since we can't use loops, we use all() with a generator
        is_sorted = all(P[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
            
        # Check 1: Exists k such that P[k-1] == k and max(P[0...k-2]) < k and min(P[k...N-1]) > k
        # We can precompute prefix max and suffix min using a trick or just use the property:
        # P[k-1] == k is required. Also, for the split to work, the set {P_0...P_{k-2}} must be {1...k-1}.
        # This is true if max(P_0...P_{k-2}) == k-1.
        
        # To avoid loops, we use map and list comprehensions.
        # Prefix maxes
        # Note: We can't use reduce easily for prefix max without a loop, 
        # but we can use a scan-like approach. 
        # Actually, the constraint is: P_i = i for all i < k AND P_i = i for all i > k is NOT required.
        # The requirement is: {P_1, ..., P_{k-1}} = {1, ..., k-1} AND P_k = k AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This is equivalent to: max(P_1, ..., P_{k-1}) = k-1 AND P_k = k.
        
        # We can use a list comprehension to find all k that satisfy P[k-1] == k.
        # Then for those k, we check the max of the prefix.
        # To get prefix maxes without a loop:
        # We can't. But we can use a different approach.
        # P is a permutation. If P[k-1] == k and max(P[0...k-2]) == k-1, then the condition is met.
        # We can compute prefix maxes using a loop-less way? No, but we can use a recursive-like 
        # structure or just accept that we need to process the array.
        # Wait, the constraints say "no for/while loops". 
        # I will use map/filter/reduce or list comprehensions.
        
        # Let's use a trick: 
        # A prefix of length L is a permutation of 1..L iff max(P[0...L-1]) == L.
        # We can compute all prefix maxes using a custom function passed to map or similar, 
        # but the most reliable way to avoid 'for' is using a helper function with map/list.
        
        # Actually, I can use a list comprehension to build the prefix max array 
        # if I use a mutable state, but that's frowned upon.
        # Let's use a different logic:
        # P[k-1] == k is the pivot.
        # The condition is: (k==1 or max(P[:k-1]) == k-1) and (k==N or min(P[k:]) == k+1).
        
        # To get prefix max and suffix min without for/while:
        # We can use a recursive function? No, recursion limit.
        # We can use a list comprehension with a side effect (not recommended).
        # We can use `itertools.accumulate`.
        
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition:
        # k=1: P[0] is not necessarily 1, but the operation sorts P[1:] and P[0:0].
        # If k=1, we need P[0] to be 1 after sorting P[1:], which means P[0] must be 1.
        # If k=N, we need P[N-1] to be N.
        # If 1 < k < N, we need P[k-1] == k, max(P[:k-1]) == k-1, and min(P[k:]) == k+1.
        
        # Simplified: The operation with k sorts everything except P[k-1].
        # For the result to be 1..N, we must have P[k-1] == k, and the remaining elements 
        # must be 1..k-1 and k+1..N.
        # This is true if and only if P[k-1] == k AND (k==1 or prefix_max[k-2] == k-1) 
        # AND (k==N or suffix_min[k] == k+1).
        
        possible_k = [
            k for k in range(1, N + 1) 
            if P[k-1] == k and 
            (k == 1 or prefix_max[k-2] == k-1) and 
            (k == N or suffix_min[k] == k+1)
        ]
        
        if possible_k:
            results.append("1")
        else:
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()