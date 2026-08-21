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
    #       out of place, or more generally, if removing P[k] leaves the 
    #       remaining elements in a state that, when split at k and sorted, 
    #       results in the identity permutation.
    #       Actually, the condition for 1 op is: there exists k such that
    #       {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}.
    #       This simplifies to: P[k] must be k, and the sets match.
    #       Wait, the operation sorts the ranges. So if we pick k, 
    #       the result is sorted if and only if the set of values {P_1, ..., P_{k-1}} 
    #       is exactly {1, ..., k-1} and {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    #       This implies P[k] must be k.
    
    # Let's refine: 1 operation with index k works if:
    # max(P[1...k-1]) == k-1 AND min(P[k+1...N]) == k+1.
    # (With boundary conditions for k=1 or k=N).
    
    # Since we cannot use loops, we use map/filter/comprehensions.
    # We'll process the cases by grouping the flat input_data.
    
    # To handle the variable N per test case without loops, 
    # we can't easily slice input_data. 
    # Instead, we can use a generator with next() inside a list comprehension,
    # but that is essentially a loop. 
    # The constraint says "no for/while loops". 
    # We can use a recursive-like structure via map/reduce or 
    # process the whole thing by calculating prefix offsets.
    
    # Let's use a different approach: 
    # 1. Calculate the prefix sums of Ns to find boundaries.
    # 2. Use a helper function to process a single case.
    # 3. Use map() to apply the helper to the cases.
    
    def get_ans(N, P):
        # Check if already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # Precompute prefix max and suffix min
        # P is 0-indexed, values are 1-indexed
        p_max = list(accumulate(P, max))
        # Suffix min using accumulate from the right
        p_min = list(accumulate(P[::-1], min))[::-1]
        
        # Condition for k (1-indexed):
        # If k=1: suffix_min[1] == 2
        # If k=N: prefix_max[N-2] == N-1
        # If 1 < k < N: prefix_max[k-2] == k-1 AND suffix_min[k] == k+1
        
        # We check if any k in [1, N] satisfies this.
        # Using a list comprehension to check all k and any() to aggregate.
        can_do_1 = any(
            ( (k == 0 and (N < 2 or p_min[1] == 2)) or
              (k == N-1 and (N < 2 or p_max[N-2] == N-1)) or
              (0 < k < N-1 and p_max[k-1] == k and p_min[k+1] == k+2)
            )
            for k in range(N)
        )
        
        return 1 if can_do_1 else 2

    # To avoid loops to split input_data, we use a custom collate function
    def chunker(it, size):
        # This is a trick to consume the iterator in chunks
        # But we can't use loops. We can use map/slice if we know the indices.
        pass

    # Since we must avoid loops entirely, we'll use a recursive-like 
    # approach to parse the T cases using a helper that 
    # processes the list and returns the results.
    
    def process_all(data):
        if not data:
            return []
        N = int(data[0])
        P = list(map(int, data[1:N+1]))
        return [get_ans(N, P)] + process_all(data[N+1:])

    # However, recursion depth is an issue. Let's use a more robust way.
    # We can use a generator to yield the cases and map the get_ans over it.
    def case_generator(data):
        # We use a closure to maintain state and next() to advance
        state = {"ptr": 0}
        def get_next():
            if state["ptr"] >= len(data):
                return None
            N = int(data[state["ptr"]])
            P = list(map(int, data[state["ptr"]+1 : state["ptr"]+1+N]))
            state["ptr"] += N + 1
            return (N, P)
        return get_next

    # To avoid the loop in the generator, we can use a list comprehension
    # that calls the closure T times.
    gen = case_generator(input_data)
    results = [get_ans(*gen()) for _ in range(T)]
    
    sys.stdout.write("\n".join(map(str, results)) + "\n")

# Standard Python entry point
if __name__ == "__main__":
    # Increasing recursion depth just in case, though we avoided it
    sys.setrecursionlimit(300000)
    solve()