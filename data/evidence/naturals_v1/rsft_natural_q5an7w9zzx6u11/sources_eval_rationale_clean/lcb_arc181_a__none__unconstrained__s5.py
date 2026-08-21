import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to process the remaining data in chunks of N + 1
    # Since we cannot use loops, we use a generator or a mapping function
    # to handle the variable length of each test case.
    
    # To avoid loops, we'll use a helper function to extract test cases
    # and map it across the range of T.
    
    # However, since N varies, we can't simply slice by a fixed width.
    # We will use a state-carrying approach via a list/iterator.
    it = iter(input_data[1:])
    
    def process_case(_):
        # Extract N
        N = int(next(it))
        # Extract the permutation P
        P = [int(next(it)) for _ in range(N)]
        
        # The problem asks for the minimum operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can find a k such that P_k = k and sorting the rest 
        # results in the identity, we need 1 op.
        # Actually, the condition for 1 operation is:
        # There exists k such that if we remove P_k, the remaining 
        # elements can be split into two sets (1 to k-1) and (k+1 to N)
        # such that all elements in the first set are < k and second are > k.
        # This is equivalent to saying: there exists k such that 
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This simplifies to: P_k = k and the set of elements before k is {1, ..., k-1}.
        
        # Let's check if 0 operations are needed:
        # P is already sorted.
        if P == sorted(P):
            return 0
        
        # Check if 1 operation is enough:
        # We need a k such that P_k = k and max(P_1...P_{k-1}) = k-1.
        # We can use prefix maximums and suffix minimums.
        # pref_max[i] = max(P_0...P_i)
        # suff_min[i] = min(P_i...P_{N-1})
        # k (1-indexed) works if P[k-1] == k and pref_max[k-2] == k-1 
        # (for k > 1) and suff_min[k] == k+1 (for k < N).
        
        # Since we can't use loops, we use list comprehensions and 
        # a trick to get prefix/suffix arrays.
        # Note: Python's itertools.accumulate is perfect for this.
        from itertools import accumulate
        
        pref_max = list(accumulate(P, max))
        # For suffix min, we reverse, accumulate, then reverse back.
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check for k in 1...N (0-indexed i = k-1)
        # Condition for i:
        # 1. P[i] == i + 1
        # 2. If i > 0, pref_max[i-1] == i
        # 3. If i < N-1, suff_min[i+1] == i + 2
        
        # We use a generator expression with 'any' to check the condition
        can_do_1 = any(
            (P[i] == i + 1 and 
             (i == 0 or pref_max[i-1] == i) and 
             (i == N-1 or suff_min[i+1] == i + 2))
            for i in range(N)
        )
        
        return 1 if can_do_1 else 2

    # Map the process_case function over the range T and print results
    results = map(process_case, range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()