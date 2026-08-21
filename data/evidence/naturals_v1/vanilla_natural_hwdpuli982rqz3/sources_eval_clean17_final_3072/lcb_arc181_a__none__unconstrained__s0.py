import sys

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
        
        # The goal is to find the minimum number of operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can pick k such that P_k is the only element not in its 
        # sorted partition, we might finish in 1 step.
        # Specifically, if there exists k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # then P_k must be k, and the array is already sorted.
        # Wait, the condition for 1 operation is:
        # There exists k such that sorting [1, k-1] and [k+1, N] results in [1, ..., N].
        # This happens if {P_1, ..., P_{k-1}} is a permutation of {1, ..., k-1}
        # AND {P_{k+1}, ..., P_N} is a permutation of {k+1, ..., N}.
        # This is equivalent to saying that for some k, 
        # max(P_1, ..., P_{k-1}) = k-1 and min(P_{k+1}, ..., P_N) = k+1.
        
        # First, check if already sorted (0 operations)
        # Since we can't use loops, we use map and list comprehensions.
        is_sorted = all(P[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
        
        # Precompute prefix maximums and suffix minimums
        # Using a trick with accumulate to avoid loops
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if there exists k (1-indexed) such that:
        # For k=1: suffix_min[1] == 2
        # For k=N: prefix_max[N-2] == N-1
        # For 1 < k < N: prefix_max[k-2] == k-1 AND suffix_min[k] == k+1
        
        # We use a list comprehension to check these conditions
        # k is 1-indexed in the problem, so k-1 is 0-indexed.
        # Let i = k-1 (0-indexed).
        # i = 0: suffix_min[1] == 2
        # i = N-1: prefix_max[N-2] == N
        # 0 < i < N-1: prefix_max[i-1] == i and suffix_min[i+1] == i+2
        
        # To handle boundaries without if/else, we can pad the arrays.
        # But since we can't use loops, we can just check the conditions in a list.
        
        # Condition for i=0
        cond0 = (suffix_min[1] == 2) if N > 1 else False
        # Condition for i=N-1
        condN = (prefix_max[N-2] == N-1) if N > 1 else False
        # Conditions for 0 < i < N-1
        condMid = any(prefix_max[i-1] == i and suffix_min[i+1] == i+2 for i in range(1, N-1))
        
        if cond0 or condN or condMid:
            results.append("1")
        else:
            # It is proved that 2 operations are always sufficient.
            # For example, k=1 then k=N.
            # k=1 sorts [2, N]. Then k=N sorts [1, N-1].
            # Actually, k=1 sorts P_2...P_N. Then P_1 is the only one left.
            # If we pick k=1, then k=N, we can sort any permutation.
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()