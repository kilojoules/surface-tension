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
    # though the constraints allow for a list comprehension here.
    
    # We need to group the input data into chunks of (N, P1...PN)
    # Since N varies, we use a custom function to parse the flat list.
    def get_cases(data):
        it = iter(data[1:])
        for _ in range(T):
            n = int(next(it))
            p = [int(next(it)) for _ in range(n)]
            yield n, p

    # The core logic:
    # An operation with index k sorts [1, k-1] and [k+1, N].
    # This means P_k remains in place, and all other elements are sorted.
    # If we can find a k such that P_k = k, and sorting the rest fixes the permutation,
    # then 1 operation suffices.
    # Sorting the rest fixes the permutation if and only if:
    # {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    # This is equivalent to saying max(P_1...P_{k-1}) = k-1.
    
    # For a given k, the condition is:
    # (k == 1 or max(P_1...P_{k-1}) == k-1) AND (k == N or min(P_{k+1}...P_N) == k+1)
    # Actually, if max(P_1...P_{k-1}) == k-1, then the first k-1 elements must be a 
    # permutation of 1...k-1. If P_k == k, then the remaining must be k+1...N.
    
    # We can check this for all k in O(N) using prefix maximums and suffix minimums.
    # If the permutation is already sorted, answer is 0.
    # If there exists k such that the condition holds, answer is 1.
    # Otherwise, the answer is 2 (it is proven that 2 operations always suffice for N >= 3).
    
    def process_case(n, p):
        # Check if already sorted
        # Using all() with a generator to maintain O(N) and short-circuit
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # Prefix maximums: pref_max[i] = max(P_0...P_i)
        pref_max = list(map(max, accumulate(p)))
        # Suffix minimums: suff_min[i] = min(P_i...P_{n-1})
        # accumulate from right to left
        suff_min = list(map(min, accumulate(p[::-1]), lambda x, y: min(x, y)))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition
        # k is 1-indexed, so index in 0-indexed array is k-1
        # For k=1: index 0. Condition: min(P_1...P_{N-1}) == 2
        # For k=N: index N-1. Condition: max(P_0...P_{N-2}) == N-1
        # For 1 < k < N: index i. Condition: pref_max[i-1] == i and suff_min[i+1] == i+2
        
        # We use a generator expression with any() for O(N) efficiency
        can_do_1 = any(
            (p[i] == i + 1 and 
             (i == 0 or pref_max[i-1] == i) and 
             (i == n-1 or suff_min[i+1] == i + 2))
            for i in range(n)
        )
        
        return 1 if can_do_1 else 2

    # Map the process_case function over the generator of cases
    results = [process_case(n, p) for n, p in get_cases(input_data)]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()