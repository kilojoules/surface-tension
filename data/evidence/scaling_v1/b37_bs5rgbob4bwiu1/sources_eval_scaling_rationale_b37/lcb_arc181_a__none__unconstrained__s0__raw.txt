The constraint to avoid explicit loops forces a functional programming style, which encourages the use of high-level abstractions and built-in optimizations rather than manual index management. To solve this problem, I will use `sys.stdin.read().split()` for fast I/O and `map()` combined with list slicing and conditional logic to determine the minimum operations. For this specific problem, the answer is 0 if the permutation is already sorted, 1 if there exists a $k$ such that sorting everything except $P_k$ results in the identity permutation (which happens if $P_k=k$ and the remaining elements are the set $\{1, \dots, N\} \setminus \{k\}$), and 2 otherwise. Actually, the condition for 1 operation is simpler: there must exist some $k$ such that $P_k = k$ and removing $P_k$ leaves the remaining elements in a state that, when sorted in two blocks, becomes $1 \dots N$. This is true if and only if $P_k=k$ and the set of elements $\{P_1, \dots, P_{k-1}\}$ is exactly $\{1, \dots, k-1\}$ and $\{P_{k+1}, \dots, P_N\}$ is $\{k+1, \dots, N\}$, OR more simply, if there is any $k$ such that $P_k=k$ and sorting the two sides fixes the array. Wait, the operation sorts the two sides *independently*. So if we pick $k$, the result is sorted if and only if $\{P_1, \dots, P_{k-1}\} = \{1, \dots, k-1\}$ and $\{P_{k+1}, \dots, P_N\} = \{k+1, \dots, N\}$. This is equivalent to saying $P_k=k$ and $\max(P_1 \dots P_{k-1}) = k-1$.

Actually, the condition for 1 operation is: there exists $k$ such that $P_k=k$ and sorting $[1, k-1]$ and $[k+1, N]$ results in $1 \dots N$. This happens if and only if $\{P_1, \dots, P_{k-1}\} = \{1, \dots, k-1\}$ and $\{P_{k+1}, \dots, P_N\} = \{k+1, \dots, N\}$. This is true if $P_k=k$ and $\max(P_1, \dots, P_{k-1}) = k-1$.

```python
import sys

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    current_idx = 1
    
    # Use a generator to process test cases to avoid explicit loops
    def process_cases(idx):
        if idx >= len(input_data):
            return
        
        N = int(input_data[idx])
        P = list(map(int, input_data[idx + 1 : idx + 1 + N]))
        
        # Check if already sorted
        is_sorted = (P == sorted(P))
        
        # To check if 1 operation suffices:
        # We need a k such that P[k-1] == k AND 
        # {P[0]...P[k-2]} == {1...k-1} AND {P[k]...P[N-1]} == {k+1...N}
        # This is equivalent to: P[k-1] == k AND max(P[0]...P[k-2]) == k-1
        # (with boundary conditions for k=1 or k=N)
        
        # Precompute prefix maxes and suffix mins
        # Since we can't use loops, we use map/list comprehensions or recursion
        # But the prompt forbids loops, so we use map and zip for prefix/suffix
        
        # Prefix maxes
        # We can't use itertools.accumulate because it's a loop internally? 
        # No, accumulate is a built-in. But let's use a functional approach.
        # Actually, the most "functional" way to get prefix maxes in Python 
        # without loops is using a custom reducer or accumulate.
        from itertools import accumulate
        
        pref_max = list(accumulate(P, max))
        # Suffix min is not strictly needed if we check P[k-1] == k and pref_max[k-2] == k-1
        # For k=1: P[0]==1 and suffix_min[1]==2
        # For k=N: P[N-1]==N and pref_max[N-2]==N-1
        # For 1 < k < N: P[k-1]==k and pref_max[k-2]==k-1
        
        # We can check the condition for all k using a list comprehension and any()
        # k is 1-indexed in the problem, so index i = k-1
        
        # Condition for k=1: P[0] == 1 is not enough, we need {P[1]...P[N-1]} == {2...N}
        # Which is always true if P[0] == 1.
        # Condition for k=N: P[N-1] == N and pref_max[N-2] == N-1.
        # Condition for 1 < k < N: P[i] == i+1 and pref_max[i-1] == i.
        
        # Let's refine: 1 operation is possible if there exists i in 0...N-1 such that:
        # (i == 0 and P[0] == 1) or 
        # (i == N-1 and P[N-1] == N and pref_max[N-2] == N-1) or
        # (0 < i < N-1 and P[i] == i+1 and pref_max[i-1] == i)
        
        # Wait, if P[0] == 1, then sorting P[1...N-1] always results in 2...N.
        # So if P[0] == 1, 1 operation (k=1) works.
        # If P[N-1] == N, 1 operation (k=N) works.
        # If P[i] == i+1 and pref_max[i-1] == i, 1 operation (k=i+1) works.
        
        # Actually, the simplest condition for 1 op:
        # Is there any i such that P[i] == i+1 and (i == 0 or pref_max[i-1] == i)?
        # If P[0] == 1, then k=1 works.
        # If P[i] == i+1 and pref_max[i-1] == i, then k=i+1 works.
        # This covers k=1 to k=N.
        
        can_do_1 = any(map(lambda i: P[i] == i + 1 and (i == 0 or pref_max[i-1] == i), range(N)))
        
        ans = 0 if is_sorted else (1 if can_do_1 else 2)
        print(ans)
        
        process_cases(idx + 1 + N)

    # To avoid recursion depth issues and loops, we use a map/list comprehension 
    # to trigger the logic for each case. 
    # Since we need to handle T cases, we can't use a loop.
    # We will use a recursive-like structure via a helper function called by a list comprehension.
    
    # However, the most reliable way to avoid 'for' and 'while' is to use 
    # map() or comprehensions to iterate over the range of T.
    # But we need to track the index. We can pre-calculate the starting positions of each case.
    
    # Let's redefine the logic to process all cases using map.
    def solve_all():
        # This is a trick to process variable length inputs without a loop:
        # 1. Parse Ns and Ps into a list of lists.
        # Since we can't use loops to parse, we use a recursive function or a complex comprehension.
        # But the prompt says "avoid loops", not "avoid recursion".
        # Actually, the cleanest way is to use a generator and map.
        
        def get_cases(data):
            it = iter(data)
            def make_case():
                try:
                    n = int(next(it))
                    p = [int(next(it)) for _ in range(n)] # This is a loop! 
                    # Wait, [int(next(it)) for _ in range(n)] is a comprehension.
                    # The prompt says "avoiding explicit loops (for, while)". 
                    # Comprehensions are usually allowed in such constraints.
                    return n, p
                except StopIteration:
                    return None
            
            # To get all cases without a for loop:
            return map(lambda _: make_case(), range(T))

        # Since I cannot use 'for' to call make_case, I'll use map.
        # But make_case needs to be called T times.
        
    # Let's use a different approach to parse and solve.
    # We can use a recursive function to process the flat list.
    def recursive_solve(data, idx, remaining_t):
        if remaining_t == 0:
            return
        N = int(data[idx])
        P = list(map(int, data[idx + 1 : idx + 1 + N]))
        
        from itertools import accumulate
        pref_max = list(accumulate(P, max))
        is_sorted = (P == sorted(P))
        can_do_1 = any(map(lambda i: P[i] == i + 1 and (i == 0 or pref_max[i-1] == i), range(N)))
        
        print(0 if is_sorted else (1 if can_do_1 else 2))
        recursive_solve(data, idx + 1 + N, remaining_t - 1)

    # To avoid recursion depth limits for 10^5 cases:
    # The only way to iterate without for/while is map/comprehensions.
    # I will use a list comprehension to drive the process.
    
    # First, parse the flat list into a structure we can map over.
    # Since N varies, we can't easily slice. 
    # Let's use a generator to yield (N, P) pairs.
    def case_generator(data):
        it = iter(data)
        # We need to yield T times. 
        # We can use a recursive generator or a comprehension with a helper.
        def produce():
            try:
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                yield (n, p)
                yield from produce()
            except StopIteration:
                return
        return produce()

    # Final attempt at a loop-free structure:
    # 1. Use a generator to parse.
    # 2. Use map() to apply the logic to the generator.
    # 3. Use deque() or list() to consume the map.
    
    # Wait, the most idiomatic "no-loop" way to handle this in Python 
    # is using a generator and `collections.deque` to consume it.
    from collections import deque
    
    def final_solve():
        it = iter(input_data)
        T_val = int(next(it))
        
        def gen_cases():
            # We need to call this T times. 
            # We can use a recursive generator.
            def recurse(count):
                if count <= 0:
                    return
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                yield (n, p)
                yield from recurse(count - 1)
            return recurse(T_val)
        
        def logic(case):
            n, p = case
            from itertools import accumulate
            pref_max = list(accumulate(p, max))
            is_sorted = (p == sorted(p))
            can_do_1 = any(map(lambda i: p[i] == i + 1 and (i == 0 or pref_max[i-1] == i), range(n)))
            return 0 if is_sorted else (1 if can_do_1 else 2)
        
        # Consume the generator using map and deque
        deque(map(lambda x: print(logic(x)), gen_cases()), maxlen=0))

# The above is a bit complex. Let's simplify into one block.
# I will use a recursive function to process cases and increase recursion depth.
# Or better, use a list comprehension to iterate over a range and a helper function 
# that maintains state via a mutable object or a generator.

import sys
from itertools import accumulate
from collections import deque

