import sys
from itertools import accumulate
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data list
    ptr = 1
    
    results = []
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find the minimum operations to sort the permutation.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If the permutation is already sorted, answer is 0.
        # If there exists a k such that sorting [1, k-1] and [k+1, N] sorts the whole array,
        # it means P[k] must be the value that ends up at position k after sorting,
        # and all elements in P[1...k-1] must be the set {1...k-1} 
        # and all elements in P[k+1...N] must be the set {k+1...N}.
        # This is equivalent to saying that for some k, 
        # max(P[1...k-1]) < P[k] < min(P[k+1...N]).
        # Since P is a permutation of 1...N, this is equivalent to:
        # max(P[1...k-1]) == k-1 AND P[k] == k AND min(P[k+1...N]) == k+1.
        
        # Check if already sorted
        # We can use a comprehension to check if all P[i] == i+1
        is_sorted = all(P[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
            
        # To check if 1 operation suffices:
        # We need a k (1-indexed) such that:
        # 1. The set {P_1, ..., P_{k-1}} is {1, ..., k-1}
        # 2. P_k = k
        # 3. The set {P_{k+1}, ..., P_N} is {k+1, ..., N}
        # Condition 1 is true if max(P_1, ..., P_{k-1}) == k-1.
        # Condition 3 is true if min(P_{k+1}, ..., P_N) == k+1.
        
        # Prefix maximums
        prefix_max = list(accumulate(P, max))
        # Suffix minimums
        # Using slice [::-1] to reverse, accumulate min, then reverse back
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # We check for k from 1 to N (0-indexed i = k-1)
        # For i=0: k=1. Prefix is empty (max=0), P[0]=1, Suffix min P[1...N-1]=2
        # For i=N-1: k=N. Prefix max P[0...N-2]=N-1, P[N-1]=N, Suffix is empty (min=N+1)
        # For 0 < i < N-1: Prefix max P[0...i-1]==i, P[i]==i+1, Suffix min P[i+1...N-1]==i+2
        
        # Using a generator expression with any() to find if such i exists
        # Handle boundaries by using 0 for empty prefix max and N+1 for empty suffix min
        can_solve_in_one = any(
            ( (prefix_max[i-1] if i > 0 else 0) == i and 
              P[i] == i + 1 and 
              (suffix_min[i+1] if i < N-1 else N + 1) == i + 2 
            )
            for i in range(N)
        )
        
        if can_solve_in_one:
            results.append("1")
        else:
            # It is proven that any permutation can be sorted in at most 2 operations.
            # For example, k=1 sorts [2, N], then k=N sorts [1, N-1].
            # Actually, k=1 sorts P[2...N]. Then P[1] is the only one out of place.
            # Then k=N sorts P[1...N-1].
            # Wait, the operation is: sort 1 to k-1 AND sort k+1 to N.
            # If we pick k=1, we sort indices 2...N. If we then pick k=N, we sort 1...N-1.
            # This combination always sorts any permutation for N >= 3.
            results.append("2")
            
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()