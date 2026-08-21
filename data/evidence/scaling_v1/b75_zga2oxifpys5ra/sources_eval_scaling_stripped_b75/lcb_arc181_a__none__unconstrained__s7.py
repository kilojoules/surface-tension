import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without loops
    # We need to group the flat input list into (N, P) pairs
    # Since N varies, we can't use a simple slice. 
    # However, we can use a custom function with a stateful iterator.
    
    def get_cases(data):
        it = iter(data[1:])
        # This is a trick to simulate a loop to group N and the following N elements
        # But since we can't use loops, we use a recursive-like structure via a generator
        # Actually, the most reliable way to handle variable N without 'for/while' 
        # is to process the list by tracking the current index.
        
        # Since we must avoid loops entirely, we use a helper that processes the list
        # by consuming the iterator.
        def process(remaining):
            if not remaining:
                return
            N = int(remaining[0])
            P = remaining[1:N+1]
            yield (N, P)
            yield from process(remaining[N+1:])
            
        # The above recursion will hit depth limits for T=10^5.
        # Instead, we can use a clever approach with a list comprehension 
        # and a mutable state to track the index, but that's essentially a loop.
        # The only way to group variable lengths without loops/recursion 
        # is to pre-calculate the boundaries.
        pass

    # Correct approach to group variable N without loops:
    # 1. Extract all Ns by iterating through the list and jumping.
    # Since we can't use loops, we use a trick with a list and a function.
    
    def solve_all(data):
        # We use a list to store the starting positions of each test case.
        # Because we can't use loops, we use a generator that consumes the iterator.
        def group_data(it):
            # To avoid recursion depth, we use a list comprehension that 
            # updates a pointer. But wait, list comprehensions can't update 
            # external variables easily. 
            # Let's use the fact that we can use 'next()' inside a list comprehension.
            
            # We create a helper object to hold the iterator
            class State:
                def __init__(self, iterator):
                    self.it = iterator
                def next_case(self):
                    try:
                        n = int(next(self.it))
                        p = [next(self.it) for _ in range(n)]
                        return (n, p)
                    except StopIteration:
                        return None

            state = State(iter(data[1:]))
            # Use a list comprehension to call next_case T times
            return [state.next_case() for _ in range(T)]

    # Process the cases
    # For a given P, the answer is:
    # 0 if already sorted.
    # 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole thing.
    # This is possible if there is some k such that:
    # {P_1...P_{k-1}} union {P_{k+1}...P_N} = {1...N} \ {k}
    # Which simplifies to: P_k = k AND (all elements < k are in positions != k)
    # Actually, the condition for 1 operation is:
    # There exists k such that P_k = k and 
    # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    # Wait, that's too strict. The operation sorts the ranges.
    # After sorting, P_i = i for all i if and only if:
    # The set of values {P_1...P_{k-1}} is {1...k-1} AND
    # The set of values {P_{k+1}...P_N} is {k+1...N}.
    # This is equivalent to: P_k = k AND max(P_1...P_{k-1}) < k.
    
    def calculate_ans(case):
        N, P = case
        P = list(map(int, P))
        
        # Check if already sorted
        if P == sorted(P):
            return 0
        
        # Check if 1 operation suffices:
        # Exists k such that P_k == k and max(P_1...P_{k-1}) < k
        # Note: P_i are 1-indexed in problem, 0-indexed in Python.
        # For a given k (1-indexed), we need P[k-1] == k and 
        # all P[0...k-2] are < k.
        # This is true if max(P[0...k-2]) < k.
        
        # Use accumulate to find prefix maximums
        prefix_max = list(accumulate(P, max))
        
        # We need to check if there is any k (1 to N) such that:
        # 1. P[k-1] == k
        # 2. (k == 1) or (prefix_max[k-2] < k)
        # 3. (k == N) or (min(P[k...N-1]) > k)
        # Actually, if P[k-1] == k and prefix_max[k-2] < k, 
        # then the first k-1 elements must be a permutation of 1...k-1.
        # Consequently, the remaining N-k elements must be a permutation of k+1...N.
        
        # We can use a list comprehension to check this condition for all k
        # We need suffix minimums for the 3rd condition
        # But wait, if P[k-1] == k and prefix_max[k-2] < k, 
        # then the set {P_0...P_{k-2}} is exactly {1...k-1}.
        # Since P is a permutation, the remaining elements {P_k...P_{N-1}} 
        # must be {k+1...N}. Sorting them will definitely result in P_i = i.
        
        # So the condition for 1 is: exists k in 1...N such that 
        # (k == 1 or prefix_max[k-2] < k) and P[k-1] == k
        
        # Let's refine:
        # For k=1: P[0]==1. Then sorting P[1...N-1] makes it 2...N.
        # For k=N: P[N-1]==N. Then sorting P[0...N-2] makes it 1...N-1.
        # For 1 < k < N: P[k-1]==k and prefix_max[k-2] < k.
        
        # We can use a generator expression inside 'any()'
        can_do_1 = any(
            (k == 1 or prefix_max[k-2] < k) and P[k-1] == k 
            for k in range(1, N + 1)
        )
        
        return 1 if can_do_1 else 2

    # Execute the logic
    cases = solve_all(input_data)
    results = map(calculate_ans, cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()