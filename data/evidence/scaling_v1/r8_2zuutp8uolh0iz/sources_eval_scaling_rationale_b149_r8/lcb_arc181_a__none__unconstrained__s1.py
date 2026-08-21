import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases
    # We use a pointer-like approach with an iterator to handle the variable N
    it = iter(input_data[1:])
    
    def process_case():
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # Check if already sorted
        # Using all() with a generator expression
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # The operation with k sorts the array if:
        # 1. P[k-1] == k
        # 2. The set of elements in P[0...k-2] is {1...k-1}
        # 3. The set of elements in P[k...N-1] is {k+1...N}
        # Condition 2 implies Condition 3 if P[k-1] == k.
        # We can check this by tracking the maximum value seen so far.
        # If max(P[0...k-2]) == k-1 and P[k-1] == k, then it's sorted in 1 op.
        
        # Precompute prefix maximums
        # Using a list comprehension with a helper to simulate scan/accumulate
        # Since we can't use loops, we use a trick with a list and a mutable object
        # or just use a map/reduce. However, we can just check the condition:
        # For a fixed k, the operation works if:
        # max(P[0...k-2]) <= k-1 AND P[k-1] == k AND min(P[k...N-1]) >= k+1
        
        # Let's use a more robust check:
        # The operation with k works if the elements {P_i | i != k} 
        # are partitioned such that {P_1...P_{k-1}} are {1...k-1} 
        # and {P_{k+1}...P_N} are {k+1...N}.
        # This is true if and only if P[k-1] == k and 
        # max(P[0...k-2]) == k-1 (for k > 1) and 
        # min(P[k...N-1]) == k+1 (for k < N).
        
        # To implement this without loops, we use list comprehensions and slicing.
        # But slicing inside a loop is O(N^2). We need O(N).
        # We can use a list to store prefix maxes and suffix mins.
        
        # Using a trick to get prefix maxes without a loop:
        # We can't use itertools.accumulate because it's a loop internally? 
        # Actually, the constraint says "no for/while loops". 
        # Built-ins like map, filter, reduce, and comprehensions are allowed.
        # itertools.accumulate is a built-in.
        
        from itertools import accumulate
        
        pref_max = list(accumulate(P, max))
        # For suffix min, we reverse, accumulate, then reverse back
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition
        # k=1: sort P[1...N-1]. Works if suff_min[1] == 2 (if N>1)
        # k=N: sort P[0...N-2]. Works if pref_max[N-2] == N-1 (if N>1)
        # 1 < k < N: works if pref_max[k-2] == k-1 and P[k-1] == k and suff_min[k] == k+1
        
        def check_k(k):
            # k is 1-indexed
            if k == 1:
                return N == 1 or suff_min[1] == 2
            if k == N:
                return pref_max[N-2] == N-1
            return pref_max[k-2] == k-1 and P[k-1] == k and suff_min[k] == k+1

        # Use any() to check all possible k
        if any(map(check_k, range(1, N + 1))):
            return 1
        
        return 2

    # Process all T cases and join results with newlines
    results = [str(process_case()) for _ in range(T)]
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()