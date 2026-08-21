import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to yield the N and P for each test case
    def get_cases(data):
        it = iter(data[1:])
        return ( (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                 for current_n in (int(next(it)) for _ in range(T)) )
    
    # Since the above generator logic is recursive/loop-like, 
    # let's use a flat list and slice it.
    
    # Re-evaluating: the constraint allows list comprehensions and map.
    # Let's process the flat list of tokens.
    
    def process():
        # We need to group the flat list into test cases.
        # Because N varies, we can't use a simple chunk.
        # However, we can use a helper function with a mutable state or 
        # just use a loop for the T cases since the constraint says 
        # "no for/while loops", but usually allows them for the top-level 
        # T loop. Wait, "no for/while loops" means NONE.
        
        # To handle variable N without loops, we can use a recursive-like 
        # structure via map or a custom reducer, but Python's recursion 
        # limit is strict. Let's use a trick with a list and an index.
        pass

# Given the strict "no loop" constraint, I will use a 
# combination of map, list comprehensions, and slicing.
# Since N is variable, I'll first parse the N values to find boundaries.

def run():
    tokens = sys.stdin.read().split()
    if not tokens:
        return
    
    T = int(tokens[0])
    
    # To avoid loops, we can use a function that processes the list 
    # using a list comprehension and a helper to track indices.
    # But since we can't use loops, we can't easily track indices.
    # Let's use a recursive-style approach via a list comprehension 
    # that builds the results.
    
    # Actually, the simplest way to handle variable N without loops 
    # is to use a generator and map.
    
    def get_all_cases():
        it = iter(tokens[1:])
        def next_case():
            try:
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                return (n, p)
            except StopIteration:
                return None
        
        # We can't use a loop to call next_case. 
        # But we can use a list comprehension with a range(T).
        # Wait, 'for _ in range(n)' is a loop. 
        # The constraint says "no for/while loops". 
        # This includes list comprehensions? 
        # "You must not use any for or while loops... 
        # List comprehensions and map/filter/reduce are allowed."
        # Okay, [int(next(it)) for _ in range(n)] is a list comprehension.
        
        return [next_case() for _ in range(T)]

    # The logic for the problem:
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that {P1...Pk-1} are the values {1...k-1} 
    #       and {Pk+1...PN} are the values {k+1...N}.
    #       This is equivalent to: there exists k such that 
    #       max(P[0...k-2]) < k < min(P[k...N-1]).
    #       Actually, the operation is: sort 1 to k-1, sort k+1 to N.
    #       The result is sorted if and only if P[k-1] == k 
    #       and {P[0...k-2]} == {1...k-1} and {P[k...N-1]} == {k+1...N}.
    #       This simplifies to: P[k-1] == k and 
    #       max(P[0...k-2]) < k and min(P[k...N-1]) > k.
    
    # Let's refine:
    # For a fixed k (1-indexed), the operation sorts P[0:k-1] and P[k:N].
    # The final array is sorted if:
    # 1. The set of elements in P[0:k-1] is {1, ..., k-1}
    # 2. P[k-1] == k
    # 3. The set of elements in P[k:N] is {k+1, ..., N}
    # This is true if and only if P[k-1] == k and 
    # (k==1 or max(P[0:k-1]) == k-1) and (k==N or min(P[k:N]) == k+1).
    
    # To check this for any k in O(N), we precompute prefix max and suffix min.
    
    def solve_case(case):
        n, p = case
        # Check if already sorted
        if p == sorted(p):
            return 0
        
        # Precompute prefix max and suffix min using a trick to avoid loops
        # We can use a list comprehension with a helper or just use the 
        # fact that we can use map/reduce.
        # Since we can't use loops, we can't easily do prefix max.
        # Wait, we can use a list comprehension to build the prefix max 
        # if we have a way to reference the previous element. 
        # That's hard without loops. 
        # But we can use `itertools.accumulate`.
        
        from itertools import accumulate
        pref_max = list(accumulate(p, max))
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check if any k works (k is 1-indexed, so index is k-1)
        # Condition: p[i] == i+1 and (i==0 or pref_max[i-1] == i) 
        #            and (i==n-1 or suff_min[i+1] == i+2)
        
        def check_k(i):
            return (p[i] == i + 1 and 
                    (i == 0 or pref_max[i-1] == i) and 
                    (i == n - 1 or suff_min[i+1] == i + 2))
        
        if any(map(check_k, range(n))):
            return 1
        
        return 2

    # Using map to apply solve_case to all cases
    results = map(solve_case, get_all_cases())
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    run()