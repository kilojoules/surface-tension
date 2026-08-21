The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise and declarative code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without loops or recursion, I will use `map` for iterating over test cases, list comprehensions for data processing, and `functools.reduce` or `itertools.accumulate` if needed. For this specific problem, the logic involves checking if the permutation is already sorted, if it can be solved in 1 operation (by finding a $k$ such that sorting $[1, k-1]$ and $[k+1, N]$ fixes everything), or otherwise requiring 2 operations.

```python
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We use a generator to handle the flat list of input data
    # to simulate the structure of T test cases.
    def get_cases(data):
        it = iter(data[1:])
        return (
            (int(next(it)), [int(next(it)) for _ in range(int(current_n))])
            for current_n in [int(next(it)) for _ in range(T)]
        )
    
    # The above generator logic is slightly flawed because it consumes it twice.
    # Let's use a more robust approach to group the flat list into test cases.
    def group_cases(data):
        # Using a helper to slice the list based on N values
        def process(idx, remaining):
            if remaining == 0:
                return []
            n = int(data[idx])
            return (n, list(map(int, data[idx+1 : idx+1+n]))) + process(idx + 1 + n, remaining - 1)
        # Since recursion is forbidden, we use a different approach to group.
        pass

    # Correct approach to group without loops or recursion:
    # We can't use recursion or loops, so we must rely on the fact that 
    # we can process the flat list using a custom function and map.
    
    # Let's redefine the logic:
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that sorting P[0:k-1] and P[k:N] results in 1..N.
    #       This is possible if there is some k such that:
    #       {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
    #       AND P_k = k.
    #       Wait, the condition is simpler: if we pick k, the elements at indices 
    #       0 to k-2 are sorted and k to N-1 are sorted.
    #       The operation makes P_i = i for all i != k. 
    #       So we need P_k = k, and the set of elements {P_i | i != k} to be {1, ..., N} \ {k}.
    #       Actually, the operation sorts the two partitions. 
    #       After sorting, P_i = i for all i < k and all i > k.
    #       The only element that doesn't move is P_k.
    #       So 1 operation is enough if there exists k such that P_k = k.
    # 2 ops: Always possible for N >= 3.
    
    # Let's refine the 1-op condition:
    # If we choose k, the elements in positions 1...k-1 become sorted, 
    # and k+1...N become sorted.
    # For the final result to be 1...N, we must have:
    # 1. The set of values {P_1, ..., P_{k-1}} must be {1, ..., k-1}
    # 2. P_k must be k
    # 3. The set of values {P_{k+1}, ..., P_N} must be {k+1, ..., N}
    # Actually, if P_k = k, then the remaining elements are exactly {1, ..., N} \ {k}.
    # When we sort the prefix and suffix, they will automatically become 1...k-1 and k+1...N.
    # So 1 operation is enough if and only if there exists at least one k such that P_k = k.
    
    # Wait, the sample 1: P = (2, 1, 3, 5, 4), N=5. 
    # P_3 = 3. So k=3 works. Output 1. Correct.
    # Sample 3: P = (3, 2, 1, 7, 5, 6, 4), N=7.
    # P_1=3, P_2=2, P_3=1, P_4=7, P_5=5, P_6=6, P_7=4.
    # P_2=2, P_5=5, P_6=6. There are k's where P_k=k. 
    # But the sample output says 2. Let me re-read.
    # "sort the 1-st through (k-1)-th terms... sort the (k+1)-th through N-th terms"
    # Sample 3: P = (3, 2, 1, 7, 5, 6, 4). 
    # If k=2, P_2=2. Sort P[1:1] and P[3:7]. 
    # P becomes (3, 2, 1, 4, 5, 6, 7). Not sorted.
    # The condition is: after sorting, P_i = i for all i.
    # This means the elements in the first partition must be the values {1, ..., k-1}
    # and the elements in the second partition must be {k+1, ..., N}.
    # This is only possible if the set of values {P_1, ..., P_{k-1}} is {1, ..., k-1}
    # AND P_k = k.
    # Which is equivalent to saying: for all i < k, P_i < k, and P_k = k.
    # Which is equivalent to saying: max(P_1, ..., P_{k-1}) < k and P_k = k.
    # Since P is a permutation, if P_k = k and max(P_1...P_{k-1}) < k, 
    # then the first k-1 elements must be a permutation of 1...k-1.
    
    # Let's re-evaluate:
    # 0 ops: P_i = i for all i.
    # 1 op: There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} AND P_k = k.
    #       (This automatically implies {P_{k+1}, ..., P_N} = {k+1, ..., N}).
    #       This is equivalent to: max(P_1, ..., P_{k-1}) = k-1 and P_k = k.
    #       For k=1, the prefix is empty (max=0), P_1=1.
    #       For k=N, the suffix is empty, P_N=N and max(P_1...P_{N-1})=N-1.
    
    # Let's check Sample 3 again: P = (3, 2, 1, 7, 5, 6, 4)
    # k=1: P_1=3 (no)
    # k=2: P_2=2, max(P_1)=3 (no)
    # k=3: P_3=1 (no)
    # k=4: P_4=7 (no)
    # k=5: P_5=5, max(3,2,1,7)=7 (no)
    # k=6: P_6=6, max(3,2,1,7,5)=7 (no)
    # k=7: P_7=4 (no)
    # None work. Answer 2. Correct.

    # Implementation using map/list comprehensions:
    def solve_case(n, p):
        # 0 ops
        if p == list(range(1, n + 1)):
            return 0
        
        # To check 1 op, we need to find if there's a k such that 
        # max(P_1...P_{k-1}) == k-1 and P_k == k.
        # We can use itertools.accumulate to get prefix maximums.
        from itertools import accumulate
        prefix_max = list(accumulate(p, max))
        
        # k is 1-indexed. 
        # For k=1: P_1 == 1
        # For k > 1: prefix_max[k-2] == k-1 and p[k-1] == k
        
        # Check k=1
        if p[0] == 1:
            # We need to check if the rest can be sorted. 
            # Actually, if P_1=1, then sorting P[2...N] will always result in 2...N.
            # So k=1 works if P_1=1.
            return 1
        
        # Check k > 1
        # We are looking for k in 2...N such that prefix_max[k-2] == k-1 and p[k-1] == k
        # This can be checked with a generator expression and 'any'
        if any(prefix_max[k-2] == k-1 and p[k-1] == k for k in range(2, n + 1)):
            return 1
            
        return 2

    # Since we can't use loops, we process the input stream.
    # The input is T, then N, then N elements, repeated T times.
    # We can use a helper to chunk the flat list.
    
    def run():
        it = iter(input_data)
        T_val = int(next(it))
        
        def get_next_case():
            try:
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                return (n, p)
            except StopIteration:
                return None

        # We can't use a loop to call get_next_case T times.
        # But we can use map with a range.
        # However, the list comprehension [int(next(it)) for _ in range(n)] 
        # is allowed as it's a comprehension.
        
        # To avoid the loop for T, we can use a recursive-like structure 
        # via map, but we need a way to pass the iterator.
        # The most reliable way is to use a list comprehension that calls a function.
        
        results = [solve_case(*get_next_case()) for _ in range(T_val)]
        sys.stdout.write('\n'.join(map(str, results)) + '\n')

    # Wait, the prompt says "no loops", and list comprehensions are technically loops.
    # But usually, in these constraints, comprehensions are allowed.
    # Let's double check. "without using any loops (for, while) or recursion".
    # If comprehensions are banned, I must use map/filter/reduce.
    # Let's rewrite to be safe.
    
    # To group the flat list into cases without loops:
    # We can't easily group variable-length chunks without a loop.
    # But we can use a custom function and map.
    
    # Let's use a different approach to parse.
    # Since sum of N is 2e5, we can't use recursion.
    # The only way to consume an iterator in chunks without for/while is 
    # to use a helper function and map, but map(func, range(T)) is essentially a loop.
    # However, the constraint says "no loops (for, while)". 
    # Usually, this implies map/filter/reduce/comprehensions are the intended tools.
    
    # Let's refine the logic into a single expression.
    pass

# Since I cannot use 'def' inside 'def' and then call it in a loop, 
# I will structure the whole thing using map and a helper.

def final_solve():
    import sys
    from itertools import accumulate
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    it = iter(input_data[1:])
    
    def process_case():
        # This function will be called by map(process_case, range(T))
        # It reads N and then N elements from the iterator.
