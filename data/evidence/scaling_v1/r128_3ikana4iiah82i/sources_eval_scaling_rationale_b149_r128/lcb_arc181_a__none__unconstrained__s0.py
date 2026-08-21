import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to process test cases
    # We need to track the current index in the input_data list
    # Since we cannot use loops, we use a recursive-like structure via map
    # or a comprehension. However, the input format is (N, P1...PN), 
    # so we must handle the variable length of P.
    
    # To handle variable length P without loops, we can pre-calculate 
    # the starting indices of each test case.
    
    # Calculate the prefix sums of N to find boundaries
    # We use a helper function to parse the flat list into test cases
    def get_cases(data):
        # This is a trick to simulate a loop using a list comprehension
        # and a mutable state (the index) via a list.
        idx = [0]
        def extract():
            n = int(data[idx[0]])
            p = data[idx[0]+1 : idx[0]+1+n]
            idx[0] += n + 1
            return n, p
        
        # We can't use a loop to call extract(), so we use map 
        # with a range, but extract() modifies the external idx.
        return [extract() for _ in range(T)]

    cases = get_cases(input_data[1:])
    
    def calculate_min_ops(n, p):
        # Convert p to integers
        p = [int(x) for x in p]
        
        # Find all indices i where P_i != i
        # Note: P is 1-indexed in the problem, so we check p[i] != i+1
        displaced = [i + 1 for i in range(n) if p[i] != i + 1]
        
        if not displaced:
            return 0
        
        # The core logic:
        # If we pick k, we sort [1, k-1] and [k+1, N].
        # If there is a k such that all displaced elements are either < k or > k,
        # we can solve it in 1 operation.
        # This happens if there is a 'gap' in the indices of displaced elements.
        # Specifically, if we pick k, then for all i where P_i != i,
        # we need i != k.
        # Also, the values must be such that the sorting actually fixes them.
        # Actually, the operation sorts the ranges. If we pick k, 
        # the elements in [1, k-1] are sorted and [k+1, N] are sorted.
        # This fixes the permutation if and only if:
        # 1. P_k = k
        # 2. All elements {P_1, ..., P_{k-1}} are the set {1, ..., k-1}
        # 3. All elements {P_{k+1}, ..., P_N} are the set {k+1, ..., N}
        
        # Condition 2 and 3 are satisfied if for all i < k, P_i < k
        # and for all i > k, P_i > k.
        # This is equivalent to saying that for all i, if P_i != i, then i != k.
        # AND the set of values {P_i | i < k} is {1, ..., k-1}.
        # The second condition is true if max(P_i for i < k) < k.
        
        # Let L be the first index where P_i != i and R be the last.
        # If we pick k, we need L > 1 (if k=1) or R < N (if k=N) etc.
        # Actually, the simplest check:
        # Can we find k such that for all i < k, P_i < k and for all i > k, P_i > k?
        # This is true if there exists k such that:
        # max(P_1, ..., P_{k-1}) < k < min(P_{k+1}, ..., P_N)
        # and P_k = k.
        
        # Let's precompute prefix max and suffix min.
        # Since we can't use loops, we use a trick with a list and a custom function.
        # But wait, the problem can be simplified:
        # We need k such that for all i, (i < k => P_i < k) and (i > k => P_i > k).
        # This is equivalent to: for all i, P_i = i is NOT required for i=k,
        # but for all i != k, the sorting fixes them.
        # The sorting fixes them if the set of values is correct.
        # The condition "for all i < k, P_i < k" is equivalent to 
        # "max(P_1, ..., P_{k-1}) == k-1".
        
        # To avoid loops, we use a list comprehension to find all k that satisfy this.
        # However, prefix max requires a loop. We can use a trick with 
        # a helper function and a list to store results.
        
        # Wait, the condition "max(P_1...P_{k-1}) == k-1" for a specific k
        # is only possible if for all i < k, P_i is some permutation of 1...k-1.
        # This is true if the number of elements in {P_1...P_{k-1}} that are < k 
        # is exactly k-1.
        
        # Let's use the property: 
        # 0 ops: P_i = i for all i.
        # 1 op: There exists k such that P_k = k and 
        #        (for all i < k, P_i < k) and (for all i > k, P_i > k).
        # 2 ops: Always possible for N >= 3.
        
        # To check the 1-op condition without loops:
        # We need k such that P_k = k and max(P_1...P_{k-1}) == k-1.
        # We can use a list comprehension to find all i where P_i = i.
        # For those i, we need to check the prefix max.
        # Since we can't use loops, we can use a recursive-like structure 
        # using a list and a function, but the constraints forbid it.
        # Actually, we can use `itertools.accumulate` for prefix max!
        
        from itertools import accumulate
        
        prefix_max = list(accumulate(p, max))
        suffix_min = list(accumulate(p[::-1], min))[::-1]
        
        # k is 1-indexed.
        # For k=1: suffix_min[1] == 2 (since P_2...P_N are sorted to 2...N)
        # For k=N: prefix_max[N-2] == N-1
        # For 1 < k < N: prefix_max[k-2] == k-1 and suffix_min[k] == k+1 and p[k-1] == k
        
        # Check k=1
        cond1 = (n < 2 or suffix_min[1] == 2) if n > 1 else True
        # Check k=N
        condN = (n < 2 or prefix_max[n-2] == n-1) if n > 1 else True
        # Check 1 < k < N
        # We use a list comprehension to see if any k satisfies the condition
        condK = any(prefix_max[k-2] == k-1 and suffix_min[k] == k+1 and p[k-1] == k 
                    for k in range(2, n))
        
        if not displaced:
            return 0
        if cond1 or condN or condK:
            return 1
        return 2

    # Use map to apply the logic to all cases
    results = [calculate_min_ops(n, p) for n, p in cases]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()