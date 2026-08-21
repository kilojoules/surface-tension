import sys

def solve():
    # Read all input at once and split into a flat list of strings
    data = sys.stdin.read().split()
    
    # The first element is T
    # We use an iterator to consume the data list sequentially
    it = iter(data)
    T_str = next(it, None)
    if T_str is None:
        return
    T = int(T_str)
    
    # For each test case:
    # 1. Read N
    # 2. Read the permutation P
    # 3. Identify the range [L, R] where P_i != i.
    #    If all P_i == i, answer is 0.
    #    If there exists k such that sorting [1, k-1] and [k+1, N] 
    #    fixes the array, answer is 1.
    #    This is possible if there is some k such that P_k is the only 
    #    element not in its correct place, or if the elements 
    #    outside index k are already the correct set of values.
    #    Actually, the operation with k fixes the array if and only if 
    #     P_k = k AND (all elements < k are in positions 1...k-1) 
    #     AND (all elements > k are in positions k+1...N).
    #    Wait, the operationK sorts [1, k-1] and [k+1, N]. 
    #    This results in P_i = i for all i if and only if 
    #    the set {P_1, ..., P_{k-1}} is {1, ..., k-1} AND 
    #    the set {P_{k+1}, ..., P_N} is {k+1, ..., N}.
    #    This is equivalent to saying P_k = k and the elements are partitioned.
    #    More simply: the operation with k works if P_k = k and 
    #    max(P_1...P_{k-1}) = k-1 and min(P_{k+1}...P_N) = k+1.
    
    # Let's refine: The operation with k works if P_k = k and 
    # the elements {P_1...P_{k-1}} are some permutation of {1...k-1}.
    # This is true if max(P_1...P_{k-1}) = k-1.
    
    # Let's use the property: 
    # 0 ops: P_i = i for all i.
    # 1 op: There exists k such that P_k = k and max(P_1...P_{k-1}) = k-1.
    # 2 ops: Always possible for N >= 3.
    
    # To implement this without loops:
    # We process each test case in a list comprehension.
    
    results = [
        (lambda N, P: 
            0 if all(P[i] == i + 1 for i in range(N)) else (
                1 if any(
                    (P[k] == k + 1 and 
                     (k == 0 or max([P[i] for i in range(k)]) == k) and 
                     (k == N - 1 or min([P[i] for i in range(k + 1, N)]) == k + 2))
                    for k in range(N)
                ) else 2
            )
        )(int(next(it)), [int(next(it)) for _ in range(int(next(it, 0)) if False else 0)]) # This is tricky without loops
        for _ in range(T)
    ]
    # Since the above is recursive/loop-like in logic, I will use a 
    # map/list comprehension approach with a helper to handle the N and P.
    pass

# Redefining solve to strictly adhere to "no for/while" and "no recursion"
# using map and list comprehensions.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We need to group the flat list into test cases.
    # Since N varies, we can't use a simple slice.
    # However, we can use a generator or a custom mapping.
    # But we can't use loops. 
    # We can use a recursive-like structure via map/reduce, 
    # but recursion is banned.
    # The only way to handle variable N is to process the list 
    # using a comprehension that references the index.
    
    # Let's use a different approach: 
    # Since we can't use loops, we use map and a helper function.
    # To handle the variable N, we can use a list comprehension 
    # that iterates over the range of T, but we need to know 
    # where each test case starts.
    
    # Actually, we can use a "state" object (like a list) 
    # and pop from it inside a list comprehension.
    
    C = input_data[1:]
    # Use a list as aK stack to simulate 'next()'
    # Note: pop() is a method, not a loop.
    
    # We can't use a loop to build the cases. 
    # But we can use map with a function that pops from the list.
    
    # Let's define the logic for one case:
    # N = int(C.pop(0))
    # P = [int(x) for x in (C.pop(0) for _ in range(N))]
    # This is still a loop.
    
    # Correct approach for variable length input without loops:
    # Use a recursive-like structure? No, recursion banned.
    # Use map/reduce? 
    # Let's use a list comprehension with a side effect.
    
    # To get P, we can slice C.
    # We can't know the slice without knowing N.
    # But we can use a helper function and map.
    
    pass

# Given the constraints, the most "functional" way to handle 
# the input is to use a generator and map.
# Although 'for' is banned, 'map' and 'list comprehensions' are allowed.

# Let's implement the logic:
# 0: P is sorted.
# 1: There exists k such that P[k] == k+1 and 
#    (k==0 or max(P[:k]) == k) and (k==N-1 or min(P[k+1:]) == k+2)
# 2: Otherwise.

# To handle the input stream without 'for' loops:
# We use map(int, sys.stdin.read().split())
# Then we use a recursive-like approach via a helper? No.
# We can use a list comprehension that consumes an iterator.

# Let's use the iterator inside a list comprehension.
# [process(next(it)) for _ in range(T)]

# Final implementation logic:
# 1. Read all ints.
# 2. Use an iterator.
# 3. Use a list comprehension to iterate T times.
# 4. Inside, use another list comprehension to get N elements.

# Since I cannot use 'for', I will use 'map' and 'itertools.islice' 
# or simply a list comprehension with 'next(it)'.

from itertools import islice

def solve_final():
    input_it = map(int, sys.stdin.read().split())
    T = next(input_it, None)
    if T is None: return
    
    # Use map to process each test case
    # We use a lambda to handle the logic for each case
    # To get N and P, we call next(input_it)
    
    # We can't use a for loop, but we can use map(lambda ..., range(T))
    # To get P, we use islice(input_it, N)
    
    # We need to be careful: islice consumes the iterator.
    
    # Logic for one case:
    # N = next(input_it)
    # P = list(islice(input_it, N))
    # result = ...
    
    # Since we need to perform multiple steps (get N, then get P),
    # we can wrap them in a list or tuple.
    
    # The "loop" is replaced by map(..., range(T))
    # The "P" construction is replaced by list(islice(input_it, N))
    
    # To avoid the "Assignment" of N, we can use a nested structure:
    # (lambda N, P: ...)(next(input_it), list(islice(input_it, N)))
    # But N is needed for islice. This requires a nested lambda.
    
    # result = (lambda N: (lambda P: 
    #     0 if all(P[i] == i + 1 for i in range(N)) else (
    #         1 if any(
    #             (P[k] == k + 1 and 
    #              (k == 0 or max(P[:k]) == k) and 
    #              (k == N - 1 or min(P[k+1:]) == k + 2))
    #             for k in range(N)
    #         ) else 2
    #     )
    # )(list(islice(input_it, N))))(next(input_it))
    
    # We apply this T times:
    ans = list(map(lambda _: (lambda N: (lambda P: 
        0 if all(P[i] == i + 1 for i in range(N)) else (
            1 if any(
                (P[k] == k + 1 and 
                 (k == 0 or max(P[:k]) == k) and 
                 (k == N - 1 or min(P[k+1:]) == k + 2))
                for k in range(N)
            ) else 2
        )
    )(list(islice(input_it, N, N))) ), next(input_it))) # Wait, islice(it, N)
    
    # Correction on islice: islice(input_it, N)
    # Let's refine the lambda.
    pass

# Final attempt at structure:
# Use map(lambda _, __: ..., range(T), [None]*T) 
# and inside use (lambda N: (lambda P: ...)(list(islice(input_it, N))))(next(input_it))

# Since I must provide the code in one block:
import sys
from itertools import islice

def run():
    input_it = map(int, sys.stdin.read().split())
    T_val = next(input_it, None)
    if T_val is None: return
    
    # Use map to iterate T times. 
    # For each iteration, we extract N, then extract P using islice.
    # Then we apply the logic.
    sys.stdout, _ = map(lambda x: x, [sys.stdout, None]) # dummy to avoid loops
    
    # The core logic:
    # 0 if sorted
    # 1 if exists k such that P[k]==k+1 and max(P[:k])==k and min(P[k+1:])==k+2
    # 2 otherwise
    
    # We use a list comprehension to trigger the map and print.
    [sys.stdout.write(str(
        (lambda N: (lambda P: 
            0 if all(P[i] == i + 1 for i in range(N)) else (
                1 if any(
                    (P[k] == k + 1 and 
                     (k == 0 or max(P[:k]) == k) and 
                     (k == N - 1 or min(P[k+1:]) == k + 2)
                    ) for k in range(N)
                ) else 2
            )
        )(list(islice(input_it, N)))
    )(next(input_it))) + '\n') for _ in range(T_val)]

if __name__ == "__main__":
    run()