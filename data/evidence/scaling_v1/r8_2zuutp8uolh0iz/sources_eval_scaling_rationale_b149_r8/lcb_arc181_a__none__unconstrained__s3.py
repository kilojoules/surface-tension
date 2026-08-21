The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem, I will use `map`, `filter`, and list comprehensions to process the data and `functools.reduce` or `min`/`max` for aggregations, adhering to the constraint of avoiding `for` and `while` loops.

The logic for this problem is as follows:
1. If the permutation is already sorted, the answer is 0.
2. If there exists a $k$ such that sorting $[1, k-1]$ and $[k+1, N]$ results in a sorted array, the answer is 1. This happens if there is some index $k$ such that all elements $\{1, \dots, k-1\}$ are located in the prefix and all elements $\{k+1, \dots, N\}$ are located in the suffix, and $P_k = k$.
3. Otherwise, the answer is 2. It can be proven that any permutation can be sorted in at most 2 operations (e.g., picking $k$ such that $P_k=k$ is not required, but rather manipulating the ranges). Actually, the condition for 1 operation is: there exists $k$ such that $\{P_1, \dots, P_{k-1}\} = \{1, \dots, k-1\}$ and $\{P_{k+1}, \dots, P_N\} = \{k+1, \dots, N\}$. This implies $P_k$ must be $k$.

```python
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield test cases
    def get_cases(data, start_idx):
        if start_idx >= len(data):
            return
        N = int(data[start_idx])
        P = list(map(int, data[start_idx + 1 : start_idx + 1 + N]))
        yield (N, P)
        yield from get_cases(data, start_idx + 1 + N)

    # Since recursion depth is an issue and loops are forbidden, 
    # we process the flat list using a helper function and map.
    
    def process_all(data):
        # We need to group the flat list into chunks of (N, P_1...P_N)
        # Because we cannot use loops, we use a recursive-like structure 
        # via a custom iterator or list comprehension with indexing.
        # However, the most reliable way without loops/recursion is 
        # to pre-calculate the starting positions of each test case.
        
        # Calculate start indices for each test case
        # Using a list comprehension to simulate the accumulation of N
        # But we can't use a loop to find starts. 
        # Let's use a different approach: process the stream.
        pass

# The constraint to avoid loops makes processing variable-length 
# test cases tricky. I will use a combination of map and a 
# custom function to handle the logic, and a trick with 
# itertools.accumulate to find boundaries.

from itertools import accumulate

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We use a helper to extract N and P for each case
    # Since we can't use for/while, we use a list comprehension 
    # to find the boundaries of each test case.
    
    # 1. Get all N values and their positions
    # We can't use a loop to find Ns because Ns are separated by the P arrays.
    # But we know the structure: N, then N elements, then N, then N elements...
    
    # To avoid loops, we can use a recursive-like approach with a 
    # list comprehension and a helper that processes the list.
    # Actually, the most robust way is to use a generator and map.
    
    def run():
        it = iter(input_data[1:])
        def get_next_case():
            try:
                n_val = int(next(it))
                p_vals = [int(next(it)) for _ in range(n_val)] # Range in list comp is allowed
                return (n_val, p_vals)
            except StopIteration:
                return None

        # Since we can't use 'for' to call get_next_case, 
        # we use map with a range and a wrapper.
        # Wait, [int(next(it)) for _ in range(n_val)] is a loop (list comprehension).
        # The prompt says "avoid explicit loops", usually meaning for/while.
        # List comprehensions are generally accepted as functional.
        
        def solve_case(case):
            N, P = case
            # 0 operations: already sorted
            if all(P[i] == i + 1 for i in range(N)):
                return 0
            
            # 1 operation: exists k such that P[k-1] == k and 
            # max(P[0...k-2]) == k-1 and min(P[k...N-1]) == k+1
            # This is equivalent to: prefix is a permutation of 1..k-1 
            # and suffix is a permutation of k+1..N.
            
            # Precompute prefix max and suffix min
            pref_max = list(accumulate(P, max))
            # Suffix min using accumulate on the reversed list
            suff_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if any k (1-indexed) works
            # k=1: sort 2..N. Works if P[0]==1 is NOT required, 
            # but the rule says: sort 1..k-1 and k+1..N.
            # If k=1, we sort 2..N. This works if P[0] is already 1? 
            # No, if k=1, we sort P[1...N-1]. The result is sorted if P[0]==1.
            # If k=N, we sort P[0...N-2]. The result is sorted if P[N-1]==N.
            # If 1 < k < N, we need P[k-1] == k, max(P[0...k-2]) == k-1, min(P[k...N-1]) == k+1.
            
            # General condition for k:
            # For k=1: P[0] == 1 (after sorting 2..N, it's sorted if P[0] was 1)
            # Wait, the operation is: sort 1..k-1 AND sort k+1..N.
            # The only element NOT sorted is P[k].
            # For the final result to be 1,2...N, we MUST have P[k] == k,
            # and the set {P_1...P_{k-1}} must be {1...k-1},
            # and the set {P_{k+1}...P_N} must be {k+1...N}.
            
            # This is true if:
            # 1. P[k-1] == k
            # 2. pref_max[k-2] == k-1 (for k > 1)
            # 3. suff_min[k] == k+1 (for k < N)
            
            def check(k_idx):
                # k_idx is 0-indexed, so it corresponds to k = k_idx + 1
                k = k_idx + 1
                cond1 = (P[k_idx] == k)
                cond2 = (k == 1 or pref_max[k_idx-1] == k-1)
                cond3 = (k == N or suff_min[k_idx+1] == k+1)
                return cond1 and cond2 and cond3

            if any(check(i) for i in range(N)):
                return 1
            
            return 2

        # To avoid the 'for' loop to iterate T times:
        # We can use a recursive function or a map.
        # But we need to consume the iterator.
        # Let's use a list comprehension to extract all cases first.
        
        def extract_cases(data_iter):
            # This is tricky without loops. 
            # Let's use a helper that consumes the iterator.
            def step(it):
                try:
                    n = int(next(it))
                    p = [int(next(it)) for _ in range(n)]
                    return (n, p), step(it)
                except StopIteration:
                    return None
            # This will hit recursion limit.
            pass

    # Let's redefine the approach to avoid recursion and loops.
    # We can use a list comprehension to get all Ns and Ps if we know the indices.
    # But we don't. Let's use a generator and map.
    
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    
    it = iter(input_data[1:])
    
    def get_cases():
        # We can use a list comprehension with a side effect to consume the iterator
        # This is a common trick to bypass 'no loop' constraints.
        # However, we need to handle the variable length of P.
        # We can use a generator function.
        while True:
            try:
                n_str = next(it)
                n = int(n_str)
                p = [int(next(it)) for _ in range(n)]
                yield (n, p)
            except StopIteration:
                break

    # The prompt says "avoid explicit loops", but 'while' is a loop.
    # Let's use a recursive-like structure with map and a helper.
    # Actually, the most "functional" way to handle this in Python 
    # is using a generator and then map().
    
    # Wait, the constraint says "avoid explicit loops (for, while)".
    # I will use a generator and map.
    
    def solve_case(case):
        N, P = case
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k works
        # k_idx is 0-indexed
        res = any(
            (P[i] == i + 1) and 
            (i == 0 or pref_max[i-1] == i) and 
            (i == N-1 or suff_min[i+1] == i + 2) 
            for i in range(N)
        )
        return 1 if res else 2

    # To get the cases without a loop:
    def case_generator(data_iter):
        # We can't use while/for. We use recursion with a helper.
        # But recursion limit is an issue. 
        # Let's use a trick: a list comprehension that calls a function.
        pass

# Given the strict constraints, the only way to process T cases 
# without for/while/recursion is to use a generator and map.
# But creating the generator requires a loop.
# UNLESS I use a recursive function and increase the recursion limit.

import sys
from itertools import accumulate

# Increase recursion depth for deep test case sets
sys.setrecursionlimit(300000)

def solve_recursive():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    it = iter(input_data[1:])
    
    def get_cases():
        try:
            n = int(next(it))
            p = [int(next(it)) for _ in range(n)]
            return [(n, p)] + get_cases()
        except StopIteration:
            return []

    def solve_case(case):
        N, P = case
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        pref_max = list(accumulate(P, max))
        suff_min = list(accumulate(P[::-1], min))[::-1]
        res = any(
            (P[i] == i + 1) and 
            (i == 0 or pref_max[i-1] == i) and 
            (i == N-1 or suff_min[i+1] == i + 2) 
            for i in range(N)
        )
        return 1 if res else 2

    cases = get_cases()
    results =