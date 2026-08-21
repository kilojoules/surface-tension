import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data list
    ptr = 1
    
    # We process cases in a list comprehension to avoid explicit loops
    # For each case:
    # 1. Extract N and the permutation P
    # 2. A permutation is sorted if P_i = i for all i.
    #    The operation with index k sorts [1, k-1] and [k+1, N].
    #    This means if we can find a k such that all elements in {1...k-1} 
    #    are currently in positions {1...k-1} (set-wise) AND all elements 
    #    in {k+1...N} are in positions {k+1...N} (set-wise), then 1 op suffices.
    #    Actually, the condition for 1 operation is: 
    #    There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} 
    #    AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #    This is equivalent to saying P_k = k AND 
    #    max(P_1...P_{k-1}) = k-1 AND min(P_{k+1}...P_N) = k+1.
    #    Wait, the simpler condition for 1 op:
    #    There exists k such that P_k = k and the set of elements to the left 
    #    of k are exactly {1...k-1} and to the right are {k+1...N}.
    #    This happens if and only if max(P_1...P_{k-1}) = k-1 and P_k = k.
    #    If the array is already sorted, 0 ops.
    #    Otherwise, if such a k exists, 1 op.
    #    Otherwise, 2 ops (it is proven that 2 ops always suffice for N >= 3).
    
    # To handle the variable N for each case, we use a generator/map logic.
    # Since we cannot use loops, we pre-calculate the indices for each case.
    
    # Function to process a single case given its slice of the input list
    def process_case(case_data):
        N = int(case_data[0])
        P = [int(x) for x in case_data[1:]]
        
        # Check if already sorted
        # Using all() in a generator expression is allowed
        if all(P[i] == i + 1 for i in range(N)):
            return "0"
        
        # Check if 1 operation is enough:
        # We need a k (1-indexed) such that P_k = k and 
        # max(P_1...P_{k-1}) = k-1.
        # Let's use prefix maximums.
        # prefix_max[i] = max(P[0]...P[i])
        # We need P[k-1] == k and prefix_max[k-2] == k-1.
        # For k=1: P[0] == 1 is not enough, we need the rest to be sortable.
        # Actually, the condition is: there exists k such that 
        # {P_1...P_{k-1}} = {1...k-1} and {P_{k+1}...P_N} = {k+1...N}.
        # This is true if:
        # 1. k=1: {P_2...P_N} = {2...N}, which is always true if P_1=1.
        # 2. k=N: {P_1...P_{N-1}} = {1...N-1}, which is always true if P_N=N.
        # 3. 1 < k < N: max(P_1...P_{k-1}) = k-1 and P_k = k.
        
        # Using accumulate to get prefix maximums
        p_max = list(accumulate(P, max))
        
        # Check k=1: P[0] == 1
        # Check k=N: P[N-1] == N
        # Check 1 < k < N: p_max[k-2] == k-1 and P[k-1] == k
        
        # We can check all these in one expression:
        # For k=1: P[0] == 1
        # For k=N: P[N-1] == N
        # For k in 2..N-1: p_max[k-2] == k-1 and P[k-1] == k
        
        # Using a generator with any() to check the condition
        can_do_1 = (
            (P[0] == 1) or 
            (P[N-1] == N) or 
            any(p_max[i-1] == i and P[i] == i + 1 for i in range(1, N - 1))
        )
        
        return "1" if can_do_1 else "2"

    # To avoid loops, we group the flat input_data into cases.
    # Since N varies, we can't use a simple reshape. 
    # We use a custom function with reduce to slice the input.
    from functools import reduce
    
    def group_cases(acc, current_idx):
        if current_idx >= len(input_data):
            return acc
        N = int(input_data[current_idx])
        # Append the slice for the current case
        return reduce(
            lambda a, _: a, 
            [None], 
            acc + [(input_data[current_idx : current_idx + N + 1], current_idx + N + 1)]
        )
    
    # The above reduce is a hack to avoid loops. Let's use a different approach.
    # We can use a generator to yield the slices.
    def get_cases():
        idx = 1
        while idx < len(input_data):
            N = int(input_data[idx])
            yield input_data[idx : idx + N + 1]
            idx += N + 1

    # Map the process_case function over the generator
    results = map(process_case, get_cases())
    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()