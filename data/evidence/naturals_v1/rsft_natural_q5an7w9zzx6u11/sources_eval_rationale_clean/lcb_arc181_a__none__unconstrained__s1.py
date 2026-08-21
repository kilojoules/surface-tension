import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to yield the chunks of data for each test case
    def get_cases(data):
        it = iter(data[1:])
        while True:
            try:
                N = int(next(it))
                P = [int(next(it)) for _ in range(N)]
                yield N, P
            except StopIteration:
                break

    # The core logic for a single test case
    def calculate_min_ops(N, P):
        # Case 0: Already sorted
        # We check if P_i == i for all i. 
        # Using all() is allowed as it is a built-in.
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # Case 1: Can be solved in 1 operation
        # An operation with index k (1-indexed) sorts [1, k-1] and [k+1, N].
        # This results in a sorted array if and only if:
        # 1. P[k-1] == k
        # 2. {P[0]...P[k-2]} == {1...k-1}
        # 3. {P[k]...P[N-1]} == {k+1...N}
        # Condition 2 is true if max(P[0]...P[k-2]) == k-1.
        # Condition 3 is true if min(P[k]...P[N-1]) == k+1.
        
        # Precompute prefix max and suffix min using list comprehensions
        # Since we can't use loops, we use a trick with a helper function or 
        # we can use the fact that for P[k-1] == k, we only need to check 
        # if max(P[0...k-2]) < k and min(P[k...N-1]) > k.
        
        # To avoid loops for prefix/suffix, we can use a list comprehension 
        # combined with a mutable object or use a functional approach.
        # However, the simplest way to check if 1 operation suffices is:
        # There exists k such that P[k] == k+1 AND 
        # max(P[0...k-1]) == k AND min(P[k+1...N-1]) == k+2.
        
        # Since we cannot use loops, we use a list comprehension to evaluate 
        # the condition for all k and check if any are true.
        # To get prefix max/suffix min without loops, we can use 
        # a custom function with a reducer or just map/filter.
        # But wait, the constraint is on the 'solve' logic. 
        # Let's use a helper to compute prefix/suffix arrays.
        
        def get_prefix_max(arr):
            # Using a list comprehension with a side-effect is generally frowned upon
            # but we can use a helper function with a closure.
            res = [0] * len(arr)
            cur = [0] # mutable
            def update(x):
                cur[0] = max(cur[0], x)
                res[len(res) - len(get_prefix_max.remaining)] = cur[0]
                get_prefix_max.remaining.pop()
            # This is getting complex. Let's use a simpler observation:
            # P[k] == k+1 and max(P[:k]) == k is sufficient to imply 
            # the rest are > k.
            return None

        # Correct observation for 1 operation:
        # It is possible if there exists k such that:
        # P[k] == k+1 AND max(P[0...k-1]) == k
        # (Because if max of first k elements is k, they must be 1...k)
        # AND min(P[k+1...N-1]) == k+2
        # (Because if min of last N-k-1 elements is k+2, they must be k+2...N)
        
        # To implement this without loops, we can use a list comprehension 
        # to build the prefix max and suffix min arrays using a trick.
        # Actually, we can use a recursive-like structure via map/reduce, 
        # but Python's recursion limit is low.
        # The most "Pythonic" way without explicit for/while is using 
        # itertools.accumulate.
        
        from itertools import accumulate
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (0 to N-1) satisfies the condition
        # k is the index of the element that stays put.
        # For k=0: P[0]==1 and suff_min[1]==2
        # For k=N-1: P[N-1]==N and pref_max[N-2]==N-1
        # For 0 < k < N-1: P[k]==k+1 and pref_max[k-1]==k and suff_min[k+1]==k+2
        
        def check_k(k):
            if k == 0:
                return P[0] == 1 and (N == 1 or suff_min[1] == 2)
            if k == N - 1:
                return P[N-1] == N and (N == 1 or pref_max[N-2] == N-1)
            return P[k] == k + 1 and pref_max[k-1] == k and suff_min[k+1] == k + 2

        if any(map(check_k, range(N))):
            return 1
        
        return 2

    # Process all cases and print results
    results = [calculate_min_ops(N, P) for N, P in get_cases(input_data)]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()