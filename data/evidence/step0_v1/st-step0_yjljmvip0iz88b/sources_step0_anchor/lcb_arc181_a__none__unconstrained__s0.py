import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to process T test cases. Since we cannot use loops, 
    # we use a helper function and map it over a range or use a list comprehension.
    # However, the input is a flat list, so we need to group it.
    # We can use a generator or a list comprehension to slice the input_data.
    
    # To avoid loops, we calculate the starting index of each test case.
    # Each test case has N followed by N elements.
    # Because N varies, we can't use a simple slice. 
    # But we can use a recursive-like structure via a list comprehension 
    # if we can pre-calculate the indices.
    # Actually, the simplest way to handle variable N without loops 
    # is to use an iterator and next().
    
    it = iter(input_data[1:])
    
    def process_case():
        # This function handles one test case
        try:
            N = int(next(it))
            # Extract the next N elements as the permutation P
            P = [int(next(it)) for _ in range(N)]
            
            # The goal is to find the minimum operations to sort P.
            # An operation with index k sorts [1, k-1] and [k+1, N].
            # If P is already sorted, answer is 0.
            # If there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array, answer is 1.
            # This happens if there is some k such that:
            # 1. All elements in P[0...k-2] are <= P[k-1] (not necessarily, but the sorted version must be)
            # Actually, the condition for 1 operation is:
            # There exists k such that the set of elements {P_1...P_{k-1}} is {1...k-1}
            # AND the set of elements {P_{k+1}...P_N} is {k+1...N}.
            # This is equivalent to saying P_k = k and the elements are partitioned correctly.
            # Wait, the operation sorts the ranges. So if we pick k, 
            # the result is sorted if and only if the set of values in P[0:k-1] is {1...k-1}
            # and the set of values in P[k:N] is {k+1...N}.
            # This implies P[k-1] must be k.
            
            # Let's refine: 
            # 0 ops: P is already sorted.
            # 1 op: There exists k such that P[k-1] == k and 
            #       max(P[0...k-2]) < k and min(P[k...N-1]) > k.
            #       (With boundaries handled: if k=1, max is 0; if k=N, min is N+1).
            # 2 ops: Always possible for N >= 3.
            
            # To check 0 ops:
            is_sorted = (P == sorted(P))
            if is_sorted:
                return 0
            
            # To check 1 op:
            # We need to find if any k exists.
            # Let pref_max[i] be max of P[0...i]
            # Let suff_min[i] be min of P[i...N-1]
            # We can't use loops to build these, so we use a trick.
            # But wait, the constraint says no for loops. 
            # We can use map/reduce or list comprehensions.
            # For pref_max and suff_min, we can use itertools.accumulate.
            from itertools import accumulate
            pref_max = list(accumulate(P, max))
            suff_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if there is any k (1-indexed) such that:
            # (k==1 or pref_max[k-2] < k) AND (P[k-1] == k) AND (k==N or suff_min[k] > k)
            # We use a list comprehension to check all k and then 'any()'
            possible_1 = any(
                ( (k == 1 or pref_max[k-2] < k) and 
                  (P[k-1] == k) and 
                  (k == N or suff_min[k] > k) )
                for k in range(1, N + 1)
            )
            
            return 1 if possible_1 else 2
        except StopIteration:
            return None

    # Use map to apply process_case T times.
    # Since process_case uses the iterator 'it', it will consume the stream.
    results = map(lambda _: process_case(), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()