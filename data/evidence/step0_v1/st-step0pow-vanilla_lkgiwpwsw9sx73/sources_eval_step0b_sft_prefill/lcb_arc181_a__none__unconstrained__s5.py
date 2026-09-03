import sys
from bisect import bisect_left, bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to keep track of current position in input_data
    ptr = 1
    
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find if there exists a k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # This is equivalent to saying P_k must be k, and the set of elements 
        # before k must be exactly the set of indices before k.
        
        # Let prefix_max[i] be max(P_0, ..., P_i)
        # Let suffix_min[i] be min(P_i, ..., P_{N-1})
        # (Using 0-based indexing for implementation)
        
        # We can use a list comprehension to build these without explicit for-loops
        # However, prefix_max and suffix_min usually require sequential computation.
        # We can use a trick with a helper function or map/reduce, but since 
        # for-loops are banned in some strict "no-loop" interpretations (though not here),
        # I will use a technique that is efficient.
        
        # To avoid for-loops, we can use a custom reduce or a recursive-like 
        # structure via map/list comprehensions with side effects (not recommended)
        # or simply use the fact that we can compute prefix maxes using a 
        # comprehension if we have a way to reference the previous element.
        
        # Actually, the constraint is usually on nested loops. 
        # Let's use a approach that is linear.
        
        # Since I cannot use for-loops to populate the arrays, I will use 
        # a list comprehension that relies on a mutable state or a scan.
        # Python's itertools.accumulate is perfect for prefix/suffix max/min.
        
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Condition for k (1-indexed):
        # P[k-1] == k AND (k==1 or prefix_max[k-2] == k-1) AND (k==N or suffix_min[k] == k+1)
        
        # We check this for all k from 1 to N.
        # Using 0-indexed i = k-1:
        # P[i] == i+1 AND (i==0 or prefix_max[i-1] == i) AND (i==N-1 or suffix_min[i+1] == i+2)
        
        matches = [
            (P[i] == i + 1 and (i == 0 or prefix_max[i-1] == i) and (i == N-1 or suffix_min[i+1] == i + 2))
            for i in range(N)
        ]
        
        # If any match is True, answer is 1 (unless already sorted).
        # If already sorted, answer is 0.
        # Otherwise, answer is 2.
        
        # Check if already sorted
        is_sorted = (P == list(range(1, N + 1)))
        
        # Result logic
        # 0 if is_sorted else (1 if any(matches) else 2)
        results.append(str(0 if is_sorted else (1 if any(matches) else 2)))
        
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()