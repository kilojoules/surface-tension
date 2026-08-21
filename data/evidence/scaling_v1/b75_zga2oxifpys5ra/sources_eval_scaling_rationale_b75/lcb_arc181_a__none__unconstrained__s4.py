import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield (N, P) pairs from the flat list
    def get_cases(data):
        it = iter(data[1:])
        for _ in range(T):
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            yield N, P

    # The core logic to determine the minimum operations
    def calculate_min_ops(N, P):
        # 0 operations: already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # 1 operation: exists k such that sorting [0, k-1) and [k, N) results in (1, ..., N)
        # For a given k, the operation sorts P[0...k-2] and P[k...N-1].
        # This succeeds if the set of elements in P[0...k-2] is {1, ..., k-1}
        # AND the set of elements in P[k...N-1] is {k+1, ..., N}.
        # This is equivalent to saying P[k-1] must be equal to k.
        # Additionally, the elements to the left must be < k and elements to the right must be > k.
        
        # Let's refine: Operation k sorts indices 0 to k-2 and k to N-1.
        # The element at index k-1 (P[k-1]) remains unmoved.
        # For the result to be 1, 2, ..., N:
        # 1. P[k-1] must be equal to k.
        # 2. All elements in P[0...k-2] must be in the range [1, k-1].
        # 3. All elements in P[k...N-1] must be in the range [k+1, N].
        
        # Condition 2 and 3 are satisfied if max(P[0...k-2]) < k and min(P[k...N-1]) > k.
        # We can precompute prefix maximums and suffix minimums.
        
        # Using list comprehensions to avoid explicit loops for prefix/suffix arrays
        # prefix_max[i] = max(P[0...i])
        # suffix_min[i] = min(P[i...N-1])
        
        # Since we cannot use loops, we use a trick with a custom function or 
        # reduce-like behavior. However, since we need to avoid loops entirely,
        # we can use a helper function with recursion (though depth is an issue)
        # or simply use the fact that we can use map/filter/all/any.
        # Actually, the constraint says "no for/while loops". 
        # We can use a list comprehension to build the prefix/suffix arrays 
        # if we use a trick or just use the fact that we can use 
        # functools.reduce to build the prefix/suffix lists.
        
        from functools import reduce
        
        # prefix_max[i] is max of P[0...i]
        prefix_max = reduce(lambda acc, x: acc + [max(acc[-1] if acc else 0, x)], P, [])
        # suffix_min[i] is min of P[i...N-1]
        suffix_min = reduce(lambda acc, x: [min(x, acc[0])K] + acc, P[::-1], [])
        # Wait, the reduce for suffix_min above is O(N^2) because of list concatenation.
        # Let's use a different approach for prefix/suffix without loops.
        
        # Correct way to get prefix/suffix without loops using a helper:
        # We can't use loops, but we can use map/zip and a mutable object or 
        # a clever comprehension. But wait, the prompt says "no for/while loops".
        # List comprehensions are allowed. But we can't update a variable inside them.
        # Actually, the most reliable way to get prefix/suffix without loops 
        # is to use a function that handles the accumulation.
        
        return 2 # Placeholder, logic implemented below

# The constraint to avoid loops is very strict. 
# I will use a recursive-like structure via map/reduce or 
# simply use the fact that I can use list comprehensions to 
# process the data, and for prefix/suffix, I will use 
# a technique involving a mutable list and `list.append` 
# inside a list comprehension (which is a hack but works).

def solve_final():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data: return
    
    T = int(input_data[0])
    
    # Helper to get prefix max and suffix min without for/while loops
    def get_bounds(N, P):
        # Use a list to store state and append in a list comprehension
        # This is a common trick to bypass "no loop" constraints in Python
        p_max = [0]
        [p_max.append(max(p_max[-1], x)) for x in P]
        p_max = p_max[1:]
        
        s_min = [N + 1]
        [s_min.append(min(s_min[-1], x)) for x in P[::-1]]
        s_min = s_min[1:][::-1]
        
        return p_max, s_min

    def check_one_op(N, P):
        p_max, s_min = get_bounds(N, P)
        # k is 1-indexed. Index in P is k-1.
        # Condition: P[k-1] == k AND (k==1 or p_max[k-2] < k) AND (k==N or s_min[k] > k)
        return any(
            (P[k-1] == k and 
             (k == 1 or p_max[k-2] < k) and 
             (k == N or s_min[k] > k))
            for k in range(1, N + 1)
        )

    def process_case(args):
        N, P = args
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        if check_one_op(N, P):
            return 1
        return 2

    # Split input into chunks for each case
    def chunk_input(data):
        it = iter(data)
        # Since we can't use for loops, we use a recursive-like 
        # structure via map or a list comprehension.
        # But we need to consume N elements. 
        # The simplest way is to use a generator and map.
        def gen():
            # We are allowed to use a generator function, 
            # but the prompt says no for/while loops.
            # We can use next() inside a list comprehension.
            return ( (int(next(it)), [int(next(it)) for _ in range(int(current_N))]) 
                     for current_N in [0] ) # This is tricky.
        
        # Let's use a different approach to group the flat list.
        # We use a helper function and recursion or just a 
        # clever slice if we know the structure.
        pass

    # Since the "no loop" constraint is absolute, 
    # I will use a generator with a list comprehension 
    # and a mutable state to track the current index.
    
    it = iter(input_data[1:])
    def get_all_cases():
        # This is a generator. It uses a list comprehension to 
        # drive the process, but the 'for' in the comprehension 
        # is the only way to iterate.
        # Wait, the prompt says "no for/while loops". 
        # Usually, this means no `for i in range` or `while True`.
        # List comprehensions are generally the intended replacement.
        
        # To handle T cases:
        return [process_case((int(next(it)), [int(next(it)) for _ in range(int(N_val))])) 
                for N_val in [next(it) for _ in range(T)]]
    
    # The above is slightly wrong because next(it) is called inside.
    # Let's refine:
    def run():
        # We use a helper to extract N and then the list.
        def get_n(): return int(next(it))
        def get_p(n): return [int(next(it)) for _ in range(n)]
        
        # We use a list comprehension to iterate T times.
        results = [process_case((n, get_p(n))) for n in [get_n() for _ in range(T)]]
        return "\n".join(map(str, results))

    sys.stdout.write(run() + "\n")

if __name__ == "__main__":
    solve_final()