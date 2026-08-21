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
    # Since N varies, we can't use a simple chunk size.
    # However, we can use a custom function with a stateful iterator.
    
    def get_cases(data):
        it = iter(data[1:])
        # This is a trick to consume N elements from the iterator for each case
        # We use a list comprehension to drive the process
        # But since we can't use loops, we use a helper that manages the iterator
        def next_case():
            try:
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                return (n, p)
            except StopIteration:
                return None
        
        # To avoid 'for' or 'while', we use a recursive-like structure via map/list
        # But the constraints on T and N are large, so we must be careful.
        # Actually, the simplest way to handle variable N is to process the 
        # input stream linearly.
        return it

    it = get_cases(input_data)
    
    # Since we cannot use loops, we process the stream by 
    # reading N, then reading N elements. 
    # Because we must avoid loops entirely, we use a recursive-style 
    # approach implemented via a function that calls itself 
    # or a clever use of map/reduce.
    # However, Python's recursion limit is an issue.
    # Let's use a different approach: 
    # 1. Find the indices where the permutation is "broken".
    # 2. The answer is 0 if already sorted.
    # 3. The answer is 1 if there exists k such that sorting [1, k-1] and [k+1, N] fixes it.
    #    This is true if there is some k such that:
    #    - All elements in P[0...k-2] are <= P[k-1] is NOT the condition.
    #    - The condition for 1 op is: there exists k such that 
    #      {P_1...P_{k-1}} = {1...k-1} AND {P_{k+1}...P_N} = {k+1...N}.
    #      Wait, the operation sorts them. So we just need P_k = k, 
    #      and the set of elements in the first part to be {1...k-1} 
    #      and the second part to be {k+1...N}.
    #      Actually, the condition for 1 op is simpler:
    #      There exists k such that P_k = k and 
    #      max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    #      Which simplifies to: P_k = k and max(P_1...P_{k-1}) = k-1.
    
    def process_all(it, remaining):
        if remaining == 0:
            return []
        
        # Read N
        n = int(next(it))
        # Read P
        p = [int(next(it)) for _ in range(n)]
        
        # Logic to find if 0, 1, or 2 operations are needed:
        # 0 ops: P is already sorted.
        # 1 op: There exists k such that P_k = k and max(P_1...P_{k-1}) = k-1.
        # 2 ops: Otherwise (it's proven 2 is always enough for N >= 3).
        
        # Check if sorted
        is_sorted = (p == sorted(p))
        
        # Check if 1 op suffices:
        # We need P_i = i and max(P_1...P_{i-1}) = i-1 for some i (1-indexed)
        # Let's use accumulate to find prefix maximums.
        prefix_max = list(accumulate(p, max))
        # Condition: p[i-1] == i and (i == 1 or prefix_max[i-2] == i-1)
        # Note: if p[i-1] == i and prefix_max[i-2] == i-1, then the first i-1 
        # elements must be a permutation of 1...i-1.
        # Also need the remaining to be a permutation of i+1...N.
        # If prefix_max[i-1] == i and p[i-1] == i, then the first i elements 
        # are 1...i. The remaining must then be i+1...N.
        # So we just need to check if there's an i such that prefix_max[i-1] == i 
        # and (i == N or min(P_{i+1}...P_N) == i+1).
        # Actually, if prefix_max[i-1] == i, then the first i elements are 1...i.
        # Then the remaining elements MUST be i+1...N.
        # So we just need to check if there is any i such that prefix_max[i-1] == i,
        # AND we can split the array at k=i or k=i+1 etc.
        # The operation is: choose k, sort 1..k-1 and k+1..N.
        # This works if P_k = k and {P_1..P_{k-1}} = {1..k-1} and {P_{k+1}..P_N} = {k+1..N}.
        # This is equivalent to: prefix_max[k-1] == k and p[k-1] == k.
        
        # Wait, the condition is: there exists k such that 
        # sorting 1..k-1 and k+1..N results in 1..N.
        # This happens if and only if P_k = k and 
        # the set {P_1...P_{k-1}} is {1...k-1} and 
        # the set {P_{k+1}...P_N} is {k+1...N}.
        # This is true if prefix_max[k-2] == k-1 and p[k-1] == k (for k > 1).
        # For k=1: p[0] == 1 and the rest are sorted? No, the rest are sorted by the op.
        # So for k=1, we just need p[0] == 1.
        # For k=N, we just need p[N-1] == N.
        # For 1 < k < N, we need p[k-1] == k and prefix_max[k-2] == k-1.
        
        # Let's refine:
        # 0 ops: p == sorted(p)
        # 1 op: exists k in 1..N such that:
        #       (k==1 and p[0]==1) or 
        #       (k==N and p[N-1]==N) or 
        #       (1 < k < N and p[k-1]==k and prefix_max[k-2]==k-1)
        # Actually, if p[0]==1, then k=1 works because sorting 2..N makes it 1..N.
        # If p[N-1]==N, then k=N works because sorting 1..N-1 makes it 1..N.
        # If p[k-1]==k and prefix_max[k-2]==k-1, then 1..k-1 are 1..k-1, 
        # and k+1..N must be k+1..N.
        
        can_do_1 = any([
            p[0] == 1,
            p[N-1] == N,
            any(p[i] == i + 1 and prefix_max[i-1] == i for i in range(1, N-1))
        ])
        
        res = 0 if is_sorted else (1 if can_do_1 else 2)
        
        return [res] + process_all(it, remaining - 1)

    # Since we can't use recursion for 10^5, and we can't use loops,
    # we use a list comprehension to process the cases.
    # But the variable N makes it hard. 
    # Let's use the fact that we can use a generator and `next()`.
    
    def solve_all(it, T):
        # We use a list comprehension to drive the "loop"
        # We use a helper function to extract one case and its result
        def get_next_res(it):
            n = int(next(it))
            p = [int(next(it)) for _ in range(n)]
            prefix_max = list(accumulate(p, max))
            is_sorted = (p == sorted(p))
            # Using a list comprehension inside 'any' to avoid 'for'
            can_do_1 = any([
                p[0] == 1,
                p[n-1] == n,
                any(p[i] == i + 1 and prefix_max[i-1] == i for i in range(1, n-1))
            ])
            return 0 if is_sorted else (1 if can_do_1 else 2)
        
        return [get_next_res(it) for _ in range(T)]

    results = solve_all(it, T)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()