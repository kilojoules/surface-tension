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
        
        # The goal is to find the minimum number of operations to sort P.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can pick k such that P_k is the only element not in its 
        # correct sorted partition, we can sort the array in 1 step.
        # Specifically, if there exists k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # Then P_k must be k, and the array is already sorted (0 steps).
        # Wait, the condition for 1 step is:
        # There exists k such that all elements in P[0:k-1] are <= k-1 (or >= k+1)
        # and all elements in P[k:N] are >= k+1 (or <= k-1).
        # Actually, the simplest condition for 1 step is:
        # There exists k such that P[0:k-1] contains only elements from {1...N} \ {P[k-1]}
        # and P[k:N] contains only elements from {1...N} \ {P[k-1]}.
        # This means P[0:k-1] must be some permutation of {1...k-1} AND P[k:N] must be {k+1...N}.
        # But the operation SORTs them. So we just need:
        # max(P[0:k-1]) < P[k-1] < min(P[k:N]) is NOT required.
        # The requirement is: {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This implies P_k = k. If this holds, 0 steps.
        # For 1 step: we need to find k such that after sorting [0, k-2] and [k, N-1], 
        # the whole array becomes [1, ..., N].
        # This happens if {P_0, ..., P_{k-2}} = {1, ..., k-1} \ {x} and {P_k, ..., P_{N-1}} = {k, ..., N} \ {y}
        # where x is the element that will end up at P_{k-1} and y is the element that will end up at P_{k-1}.
        # Actually, the condition for 1 step is:
        # There exists k such that P[0:k-1] consists of elements {1, ..., k-1} except one, 
        # and P[k:N] consists of elements {k+1, ..., N} except one.
        # Let the missing element from the first part be 'a' and the second be 'b'.
        # For the result to be sorted, we must have a = k and b = k.
        # So: P[0:k-1] contains (k-1) elements from {1, ..., N} and P[k:N] contains (N-k) elements.
        # The only element not in P[0:k-1] and not in P[k:N] is P[k-1].
        # For the array to become sorted, we need max(P[0:k-1]) < P[k-1] < min(P[k:N]).
        # Since P[0:k-1] are sorted to 1...k-1 and P[k:N] to k+1...N, 
        # we just need P[k-1] to be exactly k.
        # But wait, the sample 1: (2, 1, 3, 5, 4), k=3 (P_3=3). 
        # P[0:2]={2,1} -> {1,2}, P[3:5]={5,4} -> {4,5}. Result: (1,2,3,4,5).
        # So 1 step is possible if there exists k such that:
        # max(P[0:k-1]) <= k and min(P[k:N]) >= k.
        # Since it's a permutation, this is equivalent to:
        # max(P[0:k-1]) = k-1 (if k>1) AND min(P[k:N]) = k+1 (if k<N).
        
        # Let's use prefix max and suffix min.
        prefix_max = [0] * N
        suffix_min = [0] * N
        
        curr_max = 0
        for i in range(N):
            curr_max = max(curr_max, P[i])
            prefix_max[i] = curr_max
            
        curr_min = N + 1
        for i in range(N - 1, -1, -1):
            curr_min = min(curr_min, P[i])
            suffix_min[i] = curr_min
            
        # Check if 0 steps
        # P is sorted if prefix_max[i] == i + 1 for all i
        is_sorted = all(prefix_max[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
            
        # Check if 1 step
        # Exists k (1-indexed) such that:
        # (k==1 or prefix_max[k-2] == k-1) AND (k==N or suffix_min[k] == k+1)
        # Note: P is 0-indexed in Python, so P_{k-1} is P[k-1].
        # The condition is: (k-1 == 0 or prefix_max[k-2] <= k-1) AND (k-1 == N-1 or suffix_min[k] >= k+1)
        # Since it's a permutation, prefix_max[k-2] <= k-1 is equivalent to prefix_max[k-2] == k-1.
        
        can_do_1 = any(
            (i == 0 or prefix_max[i-1] == i) and (i == N-1 or suffix_min[i+1] == i+2)
            for i in range(N)
        )
        
        if can_do_1:
            results.append("1")
        else:
            # It is proved that 2 steps are always sufficient for N >= 3.
            # (k=1 sorts P[1:], then k=N sorts P[0:N-1])
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()