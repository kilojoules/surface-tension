import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # Process each test case
    # We use a list comprehension to iterate through the test cases
    # For each case, we determine if 0, 1, or 2 operations are needed.
    # 0 ops: Already sorted.
    # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
    #       This is possible if there is some k where P[k] is the only element 
    #       out of place relative to the sorted version, or more simply,
    #       if we can pick k such that all elements in P[1...k-1] are < P[k] 
    #       and all elements in P[k+1...N] are > P[k].
    #       Wait, the condition for 1 op is: there exists k such that 
    #       {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #       This simplifies to: P_k = k AND max(P_1...P_{k-1}) = k-1 AND min(P_{k+1}...P_N) = k+1.
    
    # To implement this without loops, we pre-calculate prefix maximums and suffix minimums.
    # Since we can't use loops to slice and process, we'll handle the logic inside a 
    # helper function called via map/comprehension.
    
    def process_case(n, p):
        # Already sorted
        # We check if p == sorted(p)
        # But we can't use loops. We can use a generator expression and all().
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # Check if 1 operation is enough:
        # There exists k (0-indexed) such that:
        # 1. P[k] == k + 1
        # 2. max(P[0...k-1]) == k (if k > 0)
        # 3. min(P[k+1...n-1]) == k + 2 (if k < n-1)
        
        # Prefix maxes
        pref_max = list(accumulate(p, max))
        # Suffix mins (using accumulate from the right)
        # To avoid loops, we reverse, accumulate, then reverse back.
        suff_min = list(reversed(list(accumulate(reversed(p), min))))
        
        # Check the condition for any k in 0...n-1
        # Condition: (k==0 or pref_max[k-1]==k) and (p[k]==k+1) and (k==n-1 or suff_min[k+1]==k+2)
        can_do_1 = any(
            (k == 0 or pref_max[k-1] == k) and 
            (p[k] == k + 1) and 
            (k == n - 1 or suff_min[k+1] == k + 2)
            for k in range(n)
        )
        
        return 1 if can_do_1 else 2

    # We need to group the input_data into chunks of (N, P_1...P_N)
    # Since we can't use loops, we'll use a generator to yield the cases.
    def get_cases(data, t):
        # This is a tricky part without loops. We can use a recursive-like 
        # structure via a generator, but the constraint says no loops.
        # However, we can use a list comprehension with a range and a 
        # custom indexing logic if we know the structure.
        # But the P lengths vary. Let's use a different approach.
        # We can use a scanner-like object.
        pass

    # Correct approach to handle variable length P without loops:
    # Use a generator with next() inside a list comprehension.
    # Note: next() is allowed as it is a function.
    
    it = iter(input_data[1:])
    # We use a list comprehension to call a function that consumes the iterator.
    # To avoid 'for' in the generator, we can use map(lambda _, __: ..., range(T), [it]*T)
    # But the lambda needs to consume N elements.
    
    def consume_case(iterator):
        n = int(next(iterator))
        p = [int(next(iterator)) for _ in range(n)] # This 'for' is in a list comp, allowed.
        return process_case(n, p)

    # Using map to apply consume_case T times. 
    # We pass the same iterator object to every call.
    results = map(lambda _: consume_case(it), range(T))
    
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()