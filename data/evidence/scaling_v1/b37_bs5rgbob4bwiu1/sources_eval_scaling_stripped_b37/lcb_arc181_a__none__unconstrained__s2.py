import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without loading everything into memory
    # though the constraints allow it, we maintain a pointer for the input list.
    ptr = 1
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The problem asks for the minimum operations to sort the permutation.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If the permutation is already sorted, answer is 0.
        # If there exists a k such that sorting [1, k-1] and [k+1, N] sorts the whole array,
        # it means P_k must be the value that ends up at position k after sorting, 
        # and all elements in P[1...k-1] must be the set {1...k-1} 
        # and P[k+1...N] must be the set {k+1...N}.
        # Actually, the condition for 1 operation is:
        # There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} 
        # AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This is equivalent to saying P_k = k AND 
        # (max(P_1...P_{k-1}) == k-1) AND (min(P_{k+1}...P_N) == k+1).
        
        # Check if already sorted
        # We can use a helper to check if P == sorted(P)
        # But we can just check if all P_i == i.
        is_sorted = all(P[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
            
        # To check if 1 operation suffices:
        # We need a k (1-indexed) such that:
        # 1. P[k-1] == k
        # 2. max(P[0...k-2]) == k-1 (if k > 1)
        # 3. min(P[k...N-1]) == k+1 (if k < N)
        
        # Prefix maximums
        pref_max = list(accumulate(P, max))
        # Suffix minimums
        # To use accumulate for suffix min, we reverse, accumulate, then reverse back
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check for k in 1...N
        # k is 1-indexed, so index in P is k-1
        # Condition: 
        # (k == 1 or pref_max[k-2] == k-1) AND 
        # (P[k-1] == k) AND 
        # (k == N or suff_min[k] == k+1)
        
        possible_1 = any(
            (i == 0 or pref_max[i-1] == i) and 
            (P[i] == i + 1) and 
            (i == N - 1 or suff_min[i+1] == i + 2)
            for i in range(N)
        )
        
        if possible_1:
            results.append("1")
        else:
            # It is proven that maximum 2 operations are always sufficient.
            # For example, k=1 sorts [2, N], then k=N sorts [1, N-1].
            # Wait, the proof says it's always possible. For N >= 3, 2 operations always work.
            # Example: k=1 sorts P[2...N]. Then P[2...N] are 2...N in order.
            # Then k=N sorts P[1...N-1]. Since P[2...N] were sorted, 
            # and we sort P[1...N-1], the only element that could be misplaced is P[1].
            # Actually, the strategy is: k=1 (sorts 2..N), then k=N (sorts 1..N-1).
            # After k=1, P[1] is some value, and P[2..N] are sorted.
            # After k=N, P[1..N-1] are sorted. Since P[2..N] were already sorted,
            # the only way it's not sorted is if the value N is at some position < N.
            # But k=1 puts N at position N. Then k=N sorts 1..N-1.
            # So 2 operations always suffice for N >= 3.
            results.append("2")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()