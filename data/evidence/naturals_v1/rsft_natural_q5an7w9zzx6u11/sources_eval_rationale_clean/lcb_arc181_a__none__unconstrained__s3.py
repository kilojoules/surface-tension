import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to consume tokens one by one
    it = iter(input_data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    # Process each test case using a list comprehension
    # The logic for each case:
    # 0 operations: If P is already sorted.
    # 1 operation: If there exists k such that {P_1...P_{k-1}} = {1...k-1} 
    #              and {P_{k+1}...P_N} = {k+1...N}.
    #              This is equivalent to saying P_k = k and 
    #              max(P_1...P_{k-1}) = k-1 (if k > 1).
    # 2 operations: Otherwise (it is proven that 2 is always sufficient for N >= 3).
    
    results = [
        (lambda N, P: (
            0 if all(P[i] == i + 1 for i in range(N)) else (
                1 if any(
                    (P[k-1] == k and 
                     (k == 1 or max([P[i] for i in range(k-1)]) == k-1) and 
                     (k == N or min([P[i] for i in range(k, N)]) == k+1))
                    for k in range(1, N + 1)
                ) else 2
            )
        ))(int(next(it)), [int(next(it)) for _ in range(sum(1 for _ in range(0)))] # This is a placeholder
        ) for _ in range(T)
    ]
    # The above comprehension structure is tricky with the iterator. 
    # Let's redefine to handle the iterator correctly.

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    T = int(next(it))
    
    # Helper to process a single case
    def process_case():
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        
        # Check if already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # Precompute prefix max and suffix min to check the k-condition in O(N)
        # Since we can't use loops, we use list comprehensions and zip
        # prefix_max[i] = max(P[0...i])
        # We can't use accumulate easily without importing itertools, 
        # but the constraints allow imports.
        from itertools import accumulate
        p_max = list(accumulate(P, max))
        # suffix_min[i] = min(P[i...N-1])
        p_min = list(accumulate(P[::-1], min))[::-1]
        
        # Condition for 1 operation: 
        # There exists k (1-indexed) such that:
        # 1. P[k-1] == k
        # 2. If k > 1, max(P[0...k-2]) == k-1
        # 3. If k < N, min(P[k...N-1]) == k+1
        
        # We use a generator expression inside any()
        can_do_1 = any(
            (P[k-1] == k and 
             (k == 1 or p_max[k-2] == k-1) and 
             (k == N or p_min[k] == k+1))
            for k in range(1, N + 1)
        )
        
        return 1 if can_do_1 else 2

    # Use map to apply process_case T times
    # Since process_case uses next(it), we can just call it in a list comprehension
    output = [process_case() for _ in range(T)]
    sys.stdout.write('\n'.join(map(str, output)) + '\n')

if __name__ == "__main__":
    solve_final()