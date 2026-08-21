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

    # For a given P, the answer is:
    # 0 if P is already sorted.
    # 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
    #   This is possible if and only if there is some k such that:
    #   {P_1, ..., P_{k-1}, P_{k+1}, ..., P_N} = {1, ..., N} \ {k}
    #   Which is always true for any k since P is a permutation.
    #   The real condition is: after sorting, P_i = i for all i.
    #   This means the set of values in positions 1...k-1 must be {1...k-1}
    #   and the set of values in positions k+1...N must be {k+1...N}.
    #   This is equivalent to saying P_k = k and the elements are partitioned.
    #   Actually, the operation sorts two blocks. The only element NOT sorted is P_k.
    #   For the result to be (1, 2, ..., N), we MUST have P_k = k, and 
    #   all elements in indices < k must be < k, and all elements in indices > k must be > k.
    #   Wait, that's just saying P_k = k and max(P_1...P_{k-1}) < k.
    #   Since it's a permutation, if P_k = k and max(P_1...P_{k-1}) < k, then 
    #   the first k-1 elements must be a permutation of 1...k-1.
    # 2 otherwise. (It is proven that 2 is always sufficient for N >= 3).
    
    def calculate_ans(N, P):
        # Check if already sorted
        # Using accumulate to check if P_i == i for all i
        # But we can just check if P == sorted(P)
        # Since we need to avoid loops, we use a generator expression with all()
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # Check if 1 operation suffices:
        # Exists k (1-indexed) such that P_k = k and 
        # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
        # Let's use accumulate to find prefix maxes and suffix mins.
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # k is 1-indexed, so index is k-1
        # For k=1: suffix_min[1] > 1 (if N > 1)
        # For k=N: prefix_max[N-2] < N (if N > 1)
        # For 1 < k < N: prefix_max[k-2] < k and suffix_min[k] > k and P[k-1] == k
        
        # We can check this in a single pass using a generator
        # Note: P is 1-indexed in problem, 0-indexed in Python list
        # For index i (0 to N-1):
        # Left side: i == 0 or prefix_max[i-1] < i + 1
        # Right side: i == N-1 or suffix_min[i+1] > i + 1
        # Middle: P[i] == i + 1
        
        can_do_1 = any(
            (i == 0 or prefix_max[i-1] < i + 1) and
            (i == N-1 or suffix_min[i+1] > i + 1) and
            (P[i] == i + 1)
            for i in range(N)
        )
        
        return 1 if can_do_1 else 2

    # Process all cases and print results
    results = [calculate_ans(N, P) for N, P in get_cases(input_data)]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()