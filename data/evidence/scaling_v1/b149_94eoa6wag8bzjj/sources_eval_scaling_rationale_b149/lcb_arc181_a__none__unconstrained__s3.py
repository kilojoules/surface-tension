import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use an iterator to consume the input_data list
    it = iter(input_data[1:])
    
    # Process each test case
    # We use a generator expression inside map to handle the logic for each case
    # The logic for each case:
    # 1. If P is already sorted, answer is 0.
    # 2. If there exists k such that sorting [1, k-1] and [k+1, N] sorts P, answer is 1.
    #    This is possible if and only if there is some k such that:
    #    - All elements in P[0...k-2] are smaller than P[k-1] (after sorting)
    #    - All elements in P[k...N-1] are larger than P[k-1] (after sorting)
    #    Actually, the condition is simpler: the operation with index k sorts the array 
    #    if and only if the element that SHOULD be at position k (which is k) 
    #    is the only element "out of place" relative to the split.
    #    More formally: the operation with k sorts P iff P[k-1] == k AND 
    #    {P_1...P_{k-1}} == {1...k-1} AND {P_{k+1}...P_N} == {k+1...N}.
    #    Wait, the operation sorts the ranges. So it succeeds if:
    #    The set of values {P_1, ..., P_{k-1}} is {1, ..., k-1} 
    #    AND the set of values {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    #    This implies P[k-1] must be k.
    
    def get_answer(N, P):
        # Check if already sorted
        # Using all() with a generator is efficient and loop-free
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # To check if there exists k such that sorting [0, k-2] and [k, N-1] sorts P:
        # We need the set of elements in P[0...k-2] to be {1...k-1} 
        # and P[k...N-1] to be {k+1...N}.
        # This is equivalent to saying that for some k, 
        # max(P[0...k-2]) < k and min(P[k...N-1]) > k.
        # Let's precompute prefix max and suffix min.
        
        # Using list comprehensions to simulate the prefix/suffix arrays
        # Since we can't use loops, we use a trick with a helper function or 
        # just realize that the condition is: 
        # there exists k such that P[k-1] == k and 
        # (k==1 or max(P[0...k-2]) == k-1) and 
        # (k==N or min(P[k...N-1]) == k+1)
        
        # To avoid loops for prefix/suffix, we can use a custom reduce or 
        # just use the fact that if P[k-1] == k and the prefix is a permutation 
        # of 1..k-1, then the suffix must be a permutation of k+1..N.
        # A prefix P[0...k-2] is a permutation of 1..k-1 iff max(P[0...k-2]) == k-1.
        
        # However, we can't use loops to build the prefix max array.
        # But we can use a list comprehension with a side-effect (not recommended)
        # or use the fact that we can use map/filter/reduce.
        # Actually, the most reliable way to do prefix/suffix without loops 
        # in Python is using itertools.accumulate.
        from itertools import accumulate
        
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition
        # k is the index (0-indexed in P is k-1)
        # Condition: (k==1 or pref_max[k-2] == k-1) and (k==N or suff_min[k] == k+1)
        # and P[k-1] == k
        
        # We use a generator expression inside any()
        possible = any(
            (P[k-1] == k and 
             (k == 1 or pref_max[k-2] == k-1) and 
             (k == N or suff_min[k] == k+1))
            for k in range(1, N + 1)
        )
        
        return 1 if possible else 2

    # Process cases
    # We use a generator to group the flat list into (N, P) pairs
    def group_cases(it):
        # This is a recursive-like structure using map/zip to avoid loops
        # But since N varies, we must consume the iterator.
        # The most "loop-free" way to handle variable length is a recursive function
        # or a comprehension that calls a function.
        return [get_answer(int(N), list(map(int, [next(it) for _ in range(int(N))])))) 
                for N in [next(it) for _ in range(T)]]

    # To avoid the list comprehension loop for N, we can use map and a helper
    def process_single_case(it):
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        return get_answer(N, P)

    # Using map to apply process_single_case T times
    results = map(lambda _: process_single_case(it), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()