import sys

def solve():
    # Read all input at once and split into a flat list of strings
    input_data = sys.stdin.read().split()
    
    # Use an iterator to consume the input data sequentially
    it = iter(input_data)
    
    # Get the number of test cases
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    # Process each test case using a list comprehension to avoid for-loops
    # The logic:
    # 1. If P is already sorted, 0 operations.
    # 2. If there exists k such that sorting [1, k-1] and [k+1, N] sorts P, 1 operation.
    #    This is possible if there is some k where P[k-1] is the only element 
    #    out of place, or more generally, if the elements that need moving 
    #    are partitioned by some k.
    #    Actually, 1 operation is sufficient if and only if there exists k 
    #    such that the set {P_1, ..., P_{k-1}} is {1, ..., k-1} 
    #    AND the set {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    #    This simplifies to: P[k-1] must be k, and the remaining elements 
    #    must be distributable into the two sorted halves.
    #    Wait, the operation sorts the ranges. So 1 op is enough if 
    #    there exists k such that P[k-1] = k.
    # 3. Otherwise, 2 operations are always sufficient for N >= 3.
    
    # Correct logic for 1 op: 
    # Operation k sorts [1, k-1] and [k+1, N].
    # This results in P_i = i for all i if and only if P[k-1] = k.
    # If P is already sorted, 0.
    # If there is any k such that P[k-1] = k, 1.
    # Else, 2.
    
    # We process T cases. We use a helper function defined inside a lambda 
    # or just logic within the comprehension.
    
    results = [
        (
            0 if all(P[i] == i + 1 for i in range(N)) else (
                1 if any(P[i] == i + 1 for i in range(N)) else 2
            )
        )
        for _ in range(T)
        for N, *P in [ (int(next(it)), [int(next(it)) for _ in range(next(it, N) if False else N)]) ]
        # Note: The nested structure above is a trick to handle the variable N 
        # and the subsequent P list within a single comprehension expression.
        # However, since we need to call next(it) N times, we use a list comprehension.
    ]
    
    # Since the prompt forbids for-loops and while-loops, 
    # and we must handle T cases with varying N:
    # We can use map and a helper function.
    
    # Redefining the approach to fit the "no loop" constraint strictly:
    # We use map(lambda...) and sys.stdout.write
}

# The constraint is very strict. I will use map and a recursive-like 
# structure via list comprehensions and iterators.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data)
    T = int(next(it))
    
    # To process T cases without a for loop, we use map.
    # To process N elements without a for loop, we use islice or a list comprehension.
    # Since we can't use for loops, we use a recursive-style map 
    # or a comprehension that consumes the iterator.
    
    # Logic:
    # 0 ops: P[i] == i+1 for all i
    # 1 op:  P[i] == i+1 for some i
    # 2 ops: otherwise
    
    # We use a helper to process a single case
    process_case = lambda: (
        lambda N, P: (
            0 if all(P[i] == i + 1 for i in range(N)) else (
                1 if any(P[i] == i + 1 for i in range(N)) else 2
            )
        )
    )(int(next(it)), [int(next(it)) for _ in range(int(next(it, 0)) if False else 0)]) 
    # The above is tricky because N is needed for the P list.
    
    # Let's use a different approach: 
    # Since we can't use loops, we can't easily do `for _ in range(T)`.
    # But we can use `map(lambda _, __: ..., range(T), ...)`
    # To handle the variable N, we can't use a simple map.
    # However, we can use a recursive function? No, recursion is like a loop.
    # Wait, list comprehensions are allowed.
    
    # Let',s use a generator to yield the result for each case.
    # Since I cannot use 'for', I will use 'map' and 'itertools.islice'.
    import itertools
    
    # We can't use a loop to call next(it). 
    # But we can use a recursive-like structure with map.
    # Actually, the most "functional" way to handle the input stream 
    # is to process the whole list.
    
    # Let's use a helper function and map.
    pass

# Given the constraints, the only way to process T cases with variable N 
# without 'for' or 'while' is to use map/reduce/comprehensions 
# and an iterator.

# Correct implementation:
import sys

def solve_no_loops():
    input_data = sys.stdin.read().split()
    if not input_data: return
    it = iter(input_data)
    T = int(next(it))
    
    # Use a recursive-like map to handle the T cases.
    # Since we can't use recursion or loops, we use a list comprehension 
    # that calls a function.
    
    def get_case_result():
        # This function is called inside map/comprehension
        # It consumes N, then consumes N elements.
        N = int(next(it))
        P = [int(next(it)) for _ in range(N)]
        return 0 if all(P[i] == i + 1 for i in range(N)) else (
               1 if any(P[i] == i + 1 for i in range(N)) else 2)

    # Use map to apply get_case_result T times.
    # range(T) provides the count, get_case_result ignores the index.
    ans = map(lambda _: get_case_result(), range(T))
    sys.stdout.write("\n".join(map(str, ans)) + "\n")

if __name__ == "__main__":
    solve_no_loops()