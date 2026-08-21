import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Use an iterator to handle the input stream without indices
    it = iter(input_data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    # The core logic for each test case:
    # 1. If already sorted, 0 operations.
    # 2. If there exists a k such that sorting [1, k-1] and [k+1, N] 
    #    results in [1, ..., N], then 1 operation.
    #    This happens if there is some k where P[k] is the only element 
    #    out of place, or more generally, if the elements that are NOT 
    #    in their correct positions are split by some index k such that
    #    all elements to the left of k are <= k and all to the right are > k.
    #    Actually, the condition for 1 operation is:
    #    There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} 
    #    AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #    This is equivalent to saying P_k = k and the remaining elements 
    #    are partitioned correctly.
    #    Wait, the operation sorts the ranges. So if we pick k, 
    #    the result is sorted if and only if the set of values {P_1, ..., P_{k-1}} 
    #    is exactly {1, ..., k-1} and {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    #    This implies P_k must be k.
    #    If P_k = k, then sorting the two sides will result in 1...N if 
    #    max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    
    # Let's refine: 
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that P_k = k and 
    #       max(P_1...P_{k-1}) = k-1 and min(P_{k+1}...P_N) = k+1.
    #       (With boundary conditions for k=1 or k=N).
    # 2 ops: Otherwise, it's always possible in 2 (as proven by problem).
    
    def get_answer(N, P):
        # Check if already sorted
        # Using all() in a generator expression is allowed
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # To check for 1 operation in O(N), we need prefix maxes and suffix mins.
        # Since we can't use loops, we use list comprehensions and a trick for 
        # prefix/suffix. However, since we can't use loops, we can't easily 
        # build a prefix max array. 
        # Wait, the constraint says "no for or while loops". 
        # We can use map, filter, reduce, and list comprehensions.
        # But we can't use a loop to build the prefix max.
        # Actually, we can use a recursive-like structure via a list comprehension 
        # if we had a way to reference previous elements, but we don't.
        # However, we can use a helper function with a mutable object or 
        # use the fact that we can use 'itertools.accumulate'.
        
        from itertools import accumulate
        
        # Prefix max: pref_max[i] = max(P[0]...P[i])
        pref_max = list(accumulate(P, max))
        # Suffix min: suff_min[i] = min(P[i]...P[N-1])
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition.
        # k is the index (0-indexed in P)
        # Condition: 
        # (k == 0 or pref_max[k-1] == k) AND 
        # (P[k] == k + 1) AND 
        # (k == N-1 or suff_min[k+1] == k + 2)
        
        # We use a generator expression inside any()
        can_do_1 = any(
            (k == 0 or pref_max[k-1] == k) and 
            (P[k] == k + 1) and 
            (k == N-1 or suff_min[k+1] == k + 2)
            for k in range(N)
        )
        
        return 1 if can_do_1 else 2

    # Process test cases
    # We use a generator to group the flat list into (N, P) pairs
    def group_cases(it, N_val):
        P = [int(next(it)) for _ in range(N_val)]
        return N_val, P

    # Since we can't use a loop to iterate T times, we use map.
    # But N varies per case, so we need a way to consume the iterator.
    # We can use a recursive function or a trick with map.
    
    def process_all(it, remaining):
        if remaining <= 0:
            return []
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        return [get_answer(N, P)] + process_all(it, remaining - 1)

    # The recursion limit might be an issue for 10^5, 
    # so let's use a different approach to avoid recursion and loops.
    # We can use a list comprehension that calls a function to consume the iterator.
    
    def solve_case():
        # This function reads N and then reads N elements.
        # Because it's called inside a list comprehension, it's allowed.
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            return get_answer(N, P)
        except StopIteration:
            return None

    # Use map(lambda _, __: solve_case(), range(T)) to execute T times.
    results = map(lambda _: solve_case(), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# Standard Python entry point
if __name__ == "__main__":
    # Increase recursion depth just in case, though we aim to avoid it.
    sys.setrecursionlimit(300000)
    solve()