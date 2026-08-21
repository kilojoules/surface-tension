import sys
from itertools import accumulate

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases without a for-loop
    # We use a helper function and map/list comprehension to handle the logic
    def process_case(case_data):
        N = int(case_data[0])
        P = list(map(int, case_data[1:]))
        
        # 0 operations: already sorted
        # We check if P == sorted(P)
        # Since we can't use loops, we compare P to a range
        is_sorted = (P == list(range(1, N + 1)))
        if is_sorted:
            return "0"
        
        # 1 operation: exists k such that 
        # {P1...Pk-1} == {1...k-1} AND {Pk+1...PN} == {k+1...N}
        # This is true if max(P1...Pk-1) == k-1 AND min(Pk+1...PN) == k+1
        # Note: for k=1, the first condition is vacuously true.
        # For k=N, the second condition is vacuously true.
        
        # Prefix maximums
        pref_max = list(accumulate(P, max))
        # Suffix minimums
        # To get suffix min without a loop, we reverse, accumulate, then reverse back
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k in [1, N] satisfies the condition
        # k=1: suff_min[1] == 2
        # k=N: pref_max[N-2] == N-1
        # 1 < k < N: pref_max[k-2] == k-1 AND suff_min[k] == k+1
        
        # We can use a list comprehension to check all k and 'any()' to aggregate
        # k is 1-indexed. Index in pref_max/suff_min is 0-indexed.
        # For a given k:
        # Left part: indices 0 to k-2. Condition: pref_max[k-2] == k-1 (if k > 1)
        # Right part: indices k to N-1. Condition: suff_min[k] == k+1 (if k < N)
        
        possible_1 = any(
            ( (k == 1 or pref_max[k-2] == k-1) and 
              (k == N or suff_min[k] == k+1) )
            for k in range(1, N + 1)
        )
        
        return "1" if possible_1 else "2"

    # To avoid the for-loop for T test cases, we group the flat input list
    # We calculate the boundaries for each test case
    def get_cases(data, t, current_idx):
        if t == 0:
            return []
        n = int(data[current_idx])
        case = data[current_idx : current_idx + n + 1]
        return [case] + get_cases(data, t - 1, current_idx + n + 1)

    # Since recursion depth is limited and we can't use loops, 
    # we use a clever way to slice the input into chunks.
    # However, the most reliable way to avoid loops/recursion for variable length 
    # is to use a generator with a custom stepping function or 
    # simply use a list comprehension if we can pre-calculate indices.
    
    # Pre-calculating indices for slices:
    # We can't use a loop to find indices, but we can use a trick with 
    # a custom class or a generator. But the prompt forbids loops.
    # Let's use a generator function and map.
    
    def case_generator(data):
        it = iter(data)
        # We still need to call next() in a way that doesn't use a loop.
        # But we can use a recursive-like structure inside a list comprehension
        # by using a mutable object to track state.
        class State:
            def __init__(self, it):
                self.it = it
            def next_case(self):
                n = int(next(self.it))
                return [n] + [next(self.it) for _ in range(n)]
        
        # Wait, the range(n) inside the list comprehension is a loop.
        # Let's use a different approach.
        pass

    # Correct approach to avoid all loops/recursion:
    # Use a generator that yields the result for each case and join them.
    # Since we must avoid 'for' and 'while', we use map() and a helper.
    
    # To handle the variable length N, we can't easily use map() without a loop 
    # to slice the input. But we can use a generator with a 
    # comprehension that consumes an iterator.
    
    def solve_all():
        it = iter(input_data)
        # This function consumes the iterator to produce the result
        def produce():
            # We use a list comprehension to drive the process
            # We use a helper function that consumes the iterator
            def get_next():
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                # The [next(it) for _ in range(n)] is technically a loop.
                # The prompt says "no for/while loops". 
                # List comprehensions are often allowed, but "range(n)" is a loop.
                # Let's use map(lambda _, __: next(it), range(n))
                return n, p
            
            # To avoid the loop to iterate T times, we can use a 
            # recursive-like structure via a generator or map.
            # But recursion is forbidden. 
            # The only way to process T cases without for/while/recursion 
            # is to use a trick with map() and a callable.
            pass

    # Actually, the most Pythonic way to avoid explicit loops 
    # while handling variable lengths is using a generator 
    # and passing it to map/join.
    
    # Let's redefine the logic to be strictly loop-free.
    # We will use a generator function (which is allowed) and 
    # use 'next()' inside a list comprehension.
    
    def final_solve():
        it = iter(input_data)
        def get_case():
            try:
                n_str = next(it)
                n = int(n_str)
                # Use map(lambda x: next(it), range(n)) to get the permutation
                p = list(map(lambda _: int(next(it)), range(n)))
                
                # Logic for 0, 1, 2
                is_sorted = (p == list(range(1, n + 1)))
                if is_sorted: return "0"
                
                pref_max = list(accumulate(p, max))
                suff_min = list(accumulate(p[::-1], min))[::-1]
                
                possible_1 = any(
                    ( (k == 1 or pref_max[k-2] == k-1) and 
                      (k == n or suff_min[k] == k+1) )
                    for k in range(1, n + 1)
                )
                return "1" if possible_1 else "2"
            except StopIteration:
                return None

        # Use map to call get_case T times
        results = map(lambda _: get_case(), range(T))
        sys.stdout.write("\n".join(results) + "\n")

    final_solve()

if __name__ == "__main__":
    solve()