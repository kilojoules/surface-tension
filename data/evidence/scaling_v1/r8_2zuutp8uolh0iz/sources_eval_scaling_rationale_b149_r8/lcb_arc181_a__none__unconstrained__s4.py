import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield the chunks of data for each test case
    def get_cases(data):
        it = iter(data[1:])
        return ( (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                 for current_n in [int(next(it)) for _ in range(T)] )
    
    # Since the above generator logic is recursive/loop-like in definition, 
    # let's use a more robust approach with a flat list and indexing.
    
    # We need to process T cases. We can use a helper function and map.
    def process_case(args):
        N = args[0]
        P = args[1:]
        
        # 0 operations: already sorted
        # We check if P_i == i for all i.
        is_sorted = all(P[i] == i + 1 for i in range(N))
        if is_sorted:
            return 0
        
        # 1 operation: exists k such that sorting [0, k-2] and [k, N-1] works.
        # This requires:
        # 1. The set of values in P[0...k-2] is {1...k-1}
        # 2. The set of values in P[k...N-1] is {k+1...N}
        # This is true if max(P[0...k-2]) == k-1 and min(P[k...N-1]) == k+1.
        
        # Precompute prefix maximums and suffix minimums
        # Using list comprehensions and a trick with a running scan via a custom function
        # Since we can't use loops, we use a list comprehension with a side-effect 
        # or a reduction. However, the cleanest way is to use a helper.
        
        # To avoid loops/recursion for prefix/suffix, we can use a trick with 
        # a list and a mutable object, but that's frowned upon. 
        # Instead, we use the fact that we can use 'itertools.accumulate'.
        from itertools import accumulate
        
        pref_max = list(accumulate(P, max))
        # For suffix min, we reverse, accumulate, then reverse back.
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check for k (1-indexed). k is the index we "leave alone".
        # k=1: only suffix [1, N-1] is sorted. Needs min(P[1:]) == 2.
        # k=N: only prefix [0, N-2] is sorted. Needs max(P[:N-1]) == N-1.
        # 1 < k < N: needs max(P[:k-1]) == k-1 AND min(P[k:]) == k+1.
        
        # We check all k from 1 to N.
        # For k=1: pref_max is empty, suff_min[1] == 2
        # For k=N: pref_max[N-2] == N-1, suff_min is empty
        # For 1 < k < N: pref_max[k-2] == k-1 and suff_min[k] == k+1
        
        # We use a generator expression inside 'any' to check the condition.
        can_do_1 = any(
            (k == 1 and suff_min[1] == 2) or
            (k == N and pref_max[N-2] == N-1) or
            (1 < k < N and pref_max[k-2] == k-1 and suff_min[k] == k+1)
            for k in range(1, N + 1)
        )
        
        return 1 if can_do_1 else 2

    # To handle the variable length of P, we parse the input stream manually.
    # Since we can't use loops, we use a recursive-like structure via map 
    # and a helper to slice the input list.
    
    def solve_all(data):
        # Use a helper to group the flat list into (N, P) pairs.
        # Since we can't use loops, we use a list comprehension that 
        # calculates the starting index of each case.
        
        # Calculate the starting position of each test case.
        # starts[i] is the index in data where case i begins.
        # We use a trick with a list and a function to simulate a scan.
        # But wait, the constraint says "no loops", but we can use 
        # comprehensions. The problem is the variable length of P.
        
        # Let's use a different approach: 
        # 1. Extract all Ns.
        # 2. Use those Ns to slice the remaining data.
        
        # This is tricky without loops. Let's use a helper function 
        # and map with a closure.
        
        all_nums = list(map(int, data))
        T = all_nums[0]
        
        # We can find the boundaries of each case using a list comprehension
        # and a trick to maintain state (using a list as a pointer).
        ptr = [1]
        def get_case():
            N = all_nums[ptr[0]]
            case_data = all_nums[ptr[0]+1 : ptr[0]+1+N]
            ptr[0] += N + 1
            return (N, case_data)
        
        # Use map to call get_case T times.
        results = map(lambda _: process_case(get_case()), range(T))
        return "\n".join(map(str, results))

    print(solve_all(input_data))

if __name__ == "__main__":
    solve()