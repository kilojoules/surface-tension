import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to group the input into test cases
    # Each case has N followed by N elements of P
    def get_cases(data):
        it = iter(data[1:])
        for _ in range(T):
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            yield N, P

    # Process each case
    # The core logic:
    # 0 ops: Already sorted.
    # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
    #       This is possible if there is some k such that:
    #       {P_1, ..., P_{k-1}, P_{k+1}, ..., P_N} = {1, ..., N} \ {k}
    #       AND the set of values in positions 1 to k-1 are all < k, 
    #       and values in positions k+1 to N are all > k.
    #       Actually, the condition for 1 op is simpler:
    #       There exists k such that P_k = k, and 
    #       max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    #       Wait, the operation sorts the two partitions. 
    #       After sorting, P_i = i for all i if and only if:
    #       The set of values {P_1, ..., P_{k-1}} is exactly {1, ..., k-1}
    #       AND P_k = k
    #       AND the set of values {P_{k+1}, ..., P_N} is exactly {k+1, ..., N}.
    # 2 ops: Always possible for N >= 3.
    
    def check_0(N, P):
        # Check if already sorted
        # Using a generator to avoid loops, though 'all' is efficient
        return all(P[i] == i + 1 for i in range(N))

    def check_1(N, P):
        # We need to find if there is a k (1-indexed) such that:
        # P_k = k AND {P_1...P_{k-1}} = {1...k-1} AND {P_{k+1}...P_N} = {k+1...N}
        # This is equivalent to:
        # For some i (0-indexed), P[i] == i + 1 AND
        # max(P[0...i-1]) <= i AND min(P[i+1...N-1]) >= i + 2
        
        # Precompute prefix maximums and suffix minimums
        # P is 0-indexed, so value k is P[i] == i + 1
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check the condition for each i
        # For i=0: P[0]==1 and suffix_min[1] >= 2
        # For i=N-1: P[N-1]==N and prefix_max[N-2] <= N-1
        # For 0 < i < N-1: P[i]==i+1 and prefix_max[i-1] <= i and suffix_min[i+1] >= i+2
        
        # Using a list comprehension to check all i and 'any' to reduce to boolean
        return any(
            (P[i] == i + 1 and 
             (i == 0 or prefix_max[i-1] <= i) and 
             (i == N-1 or suffix_min[i+1] >= i + 2))
            for i in range(N)
        )

    # Map the logic across all cases
    results = [
        0 if check_0(N, P) else (1 if check_1(N, P) else 2)
        for N, P in get_cases(input_data)
    ]
    
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()