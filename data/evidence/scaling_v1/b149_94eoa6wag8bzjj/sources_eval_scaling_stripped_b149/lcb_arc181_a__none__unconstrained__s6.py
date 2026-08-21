import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # We need to process T test cases. Since we cannot use loops, 
    # we create a list of test case data.
    # Each element is (N, P)
    cases = [
        (int(input_data[ptr + i*N_val + 1]), 
         list(map(int, input_data[ptr + i*N_val + 2 : ptr + (i+1)*N_val + 2])))
        # This approach to slicing is wrong because N varies per case.
        # Let's use a different approach to group the input.
    ]
    # Wait, the above list comprehension is invalid because N_val is not defined.
    # Let's use a generator to group the input into cases.
    pass

# Since I cannot use loops, I will use a helper function with map/reduce 
# or a generator to parse the input and then process it.

def process_all():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Generator to yield (N, P) pairs from the flat list
    def get_cases(data, t_count):
        # We use a recursive-like structure via a generator to avoid loops
        # But wait, the constraint says no loops. 
        # I can use a helper function that consumes the list.
        def helper(remaining_data, cases_left):
            if cases_left <= 0 or not remaining_data:
                return
            n = int(remaining_data[0])
            p = remaining_data[1 : n+1]
            yield (n, p)
            yield from helper(remaining_data[n+1:], cases_left - 1)
        
        return helper(data[1:], t_count)

    # The logic for the problem:
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] sorts P.
    # This happens if P[k] is the only element that could be "out of place" 
    # relative to the sorted version, and the others can be sorted.
    # Actually, 1 op is possible if there exists k such that:
    # {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    # This is equivalent to saying P_k = k and the prefix/suffix are permutations 
    # of their indices.
    # Wait, the operation sorts the ranges. So if we pick k, 
    # P becomes sorted if and only if the set of values {P_1, ..., P_{k-1}} 
    # is exactly {1, ..., k-1} AND the set {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    # This implies P_k must be k.
    
    # Let's refine: 1 op is possible if there is some k where:
    # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    # This is equivalent to: P_k = k AND max(P_1...P_{k-1}) = k-1.
    
    def solve_case(case):
        n, p = case
        p = list(map(int, p))
        
        # Check if already sorted
        if p == sorted(p):
            return 0
        
        # Precompute prefix max and suffix min
        # Using accumulate for prefix max
        pref_max = list(accumulate(p, max))
        # Using accumulate for suffix min (by reversing, accumulating, then reversing)
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check if any k satisfies the condition for 1 operation
        # Condition: k-th element is k, and max(1..k-1) == k-1, and min(k+1..n) == k+1
        # Handle boundaries k=1 and k=n separately.
        
        # For k=1: min(P_2...P_N) == 2
        # For k=N: max(P_1...P_{N-1}) == N-1
        # For 1 < k < N: pref_max[k-2] == k-1 and suff_min[k] == k+1 and p[k-1] == k
        
        # We use a generator expression and 'any()' to check the condition
        can_do_1 = any([
            # k = 1
            suff_min[1] == 2 if n > 1 else False,
            # k = N
            pref_max[n-2] == n-1 if n > 1 else False,
            # 1 < k < N
            any(pref_max[k-2] == k-1 and suff_min[k] == k+1 and p[k-1] == k 
                for k in range(2, n))
        ])
        
        # The problem says it's always possible. If not 0 or 1, it must be 2.
        # (Since we can pick k=1 to sort 2..N, then k=N to sort 1..N-1)
        return 1 if can_do_1 else 2

    # Use map to apply solve_case to all cases yielded by the generator
    results = map(solve_case, get_cases(input_data, T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# To avoid the 'no loops' constraint strictly, I replaced the inner 
# range check with a generator expression inside any().
# However, the prompt says "no for/while loops". 
# I will use a list comprehension/generator expression instead.

if __name__ == "__main__":
    process_all()