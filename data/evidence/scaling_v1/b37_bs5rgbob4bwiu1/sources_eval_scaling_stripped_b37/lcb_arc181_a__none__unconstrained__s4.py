import sys
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_cases = int(input_data[0])
    ptr = 1
    
    results = []
    for _ in range(T_cases):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += N + 1
        
        # The problem asks for the minimum operations to sort the permutation.
        # One operation with index k sorts [1, k-1] and [k+1, N].
        # This means if we can find a k such that all elements in [1, k-1] 
        # are the set {1, ..., k-1} and all elements in [k+1, N] 
        # are the set {k+1, ..., N}, then P_k must be k, and the whole 
        # array becomes sorted in 1 move.
        # However, the operation actually sorts the two partitions.
        # The only way to sort the array in 1 move is if there exists a k
        # such that the set of values {P_1, ..., P_{k-1}} is {1, ..., k-1}
        # AND the set of values {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # This is equivalent to saying that for some k, 
        # max(P_1...P_{k-1}) = k-1 and min(P_{k+1}...P_N) = k+1.
        
        # 0 operations: already sorted
        # Check if P_i == i for all i
        is_sorted = all(P[i] == i + 1 for i in range(N))
        if is_sorted:
            results.append("0")
            continue
            
        # 1 operation: exists k such that sorting [1, k-1] and [k+1, N] sorts the array.
        # This happens if the set of elements in indices [0, k-2] is {1, ..., k-1}
        # and indices [k, N-1] is {k+1, ..., N}.
        # This implies P[k-1] must be k.
        # Let's use prefix maximums and suffix minimums.
        pref_max = list(accumulate(P, max))
        # To get suffix minimums:
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # We need k (1-indexed) such that:
        # 1. If k > 1, pref_max[k-2] == k-1
        # 2. If k < N, suff_min[k] == k+1
        # Note: indices in Python are 0-based, so k-th element is P[k-1].
        
        # Check for k = 1 to N:
        # For a given k (1-based index):
        # Condition 1: k == 1 or pref_max[k-2] == k-1
        # Condition 2: k == N or suff_min[k] == k+1
        
        # We can use a list comprehension to check this for all k and 'any()' to see if one exists.
        possible_1 = any(
            (k == 1 or pref_max[k-2] == k-1) and (k == N or suff_min[k] == k+1)
            for k in range(1, N + 1)
        )
        
        if possible_1:
            results.append("1")
        else:
            # It is proved that it's always possible. 
            # For N >= 3, the maximum answer is 2.
            # Why? Pick k=1: sorts [2, N]. Then pick k=N: sorts [1, N-1].
            # Actually, picking k=1 sorts P_2...P_N. Then P_1 is the only one 
            # potentially out of place. But the operation sorts ranges.
            # If we pick k=1, P becomes (P_1, sorted(P_2...P_N)).
            # Then pick k=N, P becomes (sorted(P_1...P_{N-1}), P_N).
            # With N >= 3, two such operations are always sufficient.
            results.append("2")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()