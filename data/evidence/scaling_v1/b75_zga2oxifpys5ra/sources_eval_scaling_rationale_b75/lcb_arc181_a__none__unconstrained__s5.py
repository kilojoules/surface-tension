import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We use a generator to group the flat list into (N, P) pairs
    # Since we cannot use loops, we use a list comprehension to process all test cases
    # and map a logic function over the grouped data.
    
    # To handle the variable length of P for each T, we can't use a simple slice.
    # Instead, we process the input stream by tracking indices.
    # However, since we must avoid loops, we can use a recursive-like structure 
    # via a helper function and map, but the prompt forbids recursion.
    # The only way to handle variable length N without loops or recursion 
    # is to pre-calculate the boundaries or use a clever trick with 
    # itertools.accumulate to find the start of each test case.
    
    from itertools import accumulate
    
    # Extract N values to find boundaries
    # We need to find where each test case starts.
    # Let's use a approach where we parse the list manually using a comprehension.
    # Since we can't loop, we can't easily maintain a pointer.
    # But we can use a list comprehension to build a list of (N, P) tuples
    # by calculating the cumulative sum of Ns.
    
    # Actually, the simplest way to avoid loops/recursion while handling 
    # variable length input is to use a generator and next().
    # But next() inside a comprehension is effectively a loop.
    # Let's use a different approach: 
    # 1. Get all tokens.
    # 2. Use a custom function with a closure or a mutable object to track index.
    
    def get_cases(data):
        # Using a list to simulate a mutable pointer
        ptr = [1]
        def extract():
            n = int(data[ptr[0]])
            p = data[ptr[0]+1 : ptr[0]+1+n]
            ptr[0] += n + 1
            return n, p
        
        # We still need to call extract T times. 
        # We can use map(lambda _, __: extract(), range(T))
        return map(lambda _: extract(), range(T))

    # The logic for a single case:
    # The answer is 0 if already sorted.
    # The answer is 1 if there exists k such that sorting [1, k-1] and [k+1, N] 
    # results in [1, ..., N].
    # This happens if there is some k such that:
    # All elements in P[0:k-1] are <= k-1 (after sorting) AND
    # All elements in P[k:N] are >= k+1 (after sorting)
    # Wait, the operation sorts them. So we just need:
    # The set of elements in P[0:k-1] must be {1, ..., k-1}
    # AND the set of elements in P[k:N] must be {k+1, ..., N}
    # This is equivalent to saying P[k-1] (the k-th element) must be k.
    # AND the max of the first k-1 elements must be k-1.
    # AND the min of the last N-k elements must be k+1.
    
    def calculate_ans(n, p):
        p = [int(x) for x in p]
        # Check if already sorted
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # Check if 1 operation is enough:
        # Exists k (1-indexed) such that:
        # Sorted(P[0...k-2]) == 1...k-1 AND Sorted(P[k...n-1]) == k+1...n
        # This means:
        # 1. P[k-1] == k
        # 2. max(P[0...k-2]) == k-1 (if k > 1)
        # 3. min(P[k...n-1]) == k+1 (if k < n)
        
        # Precompute prefix max and suffix min
        # Since we can't use loops, we use a trick with a helper function 
        # or just use the fact that we can use list comprehensions.
        # But prefix max requires the previous value. 
        # We can use a trick: since we can't use loops, we can't use reduce/accumulate 
        # for complex logic? No, accumulate is allowed.
        from itertools import accumulate
        
        pref_max = list(accumulate(p, max))
        # For suffix min, we reverse, accumulate, then reverse back
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check for k in 1...N
        # k=1: P[0]==1 and suff_min[1]==2
        # k=N: P[N-1]==N and pref_max[N-2]==N-1
        # 1 < k < N: P[k-1]==k and pref_max[k-2]==k-1 and suff_min[k]==k+1
        
        # We can check this using a generator expression inside 'any()'
        can_do_1 = any(
            (p[k-1] == k and 
             (k == 1 or pref_max[k-2] == k-1) and 
             (k == n or suff_min[k] == k+1))
            for k in range(1, n + 1)
        )
        
        return 1 if can_do_1 else 2

    # Execute the logic
    cases = get_cases(input_data)
    results = map(lambda x: calculate_ans(*x), cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()