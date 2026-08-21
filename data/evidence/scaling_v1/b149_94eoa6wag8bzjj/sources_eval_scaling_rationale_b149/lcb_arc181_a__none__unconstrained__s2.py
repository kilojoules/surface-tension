The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise and declarative code by focusing on "what" to compute rather than "how" to iterate. For this problem, I will use list comprehensions for data processing and `functools.reduce` or `map` to handle transformations, and `sys.stdin.read().split()` to ingest all input at once for efficiency.

The logic to solve this problem is as follows:
1. If the permutation is already sorted, the answer is 0.
2. If there exists a $k$ such that sorting $[1, k-1]$ and $[k+1, N]$ results in a sorted array, the answer is 1. This happens if there is some index $k$ where all elements to the left of $k$ are smaller than $P_k$ and all elements to the right are larger, OR more simply, if removing $P_k$ leaves the remaining elements in their correct relative sorted positions. Specifically, the operation with $k$ succeeds if the set $\{P_1, \dots, P_{k-1}\}$ is exactly $\{1, \dots, k-1\}$ and $\{P_{k+1}, \dots, P_N\}$ is $\{k+1, \dots, N\}$. This is equivalent to saying $P_k = k$ and the prefix/suffix are permutations of their respective ranges. Wait, the operation sorts the ranges. So if we pick $k$, the result is sorted if and only if the set of values $\{P_1, \dots, P_{k-1}\} \cup \{P_{k+1}, \dots, P_N\}$ is $\{1, \dots, N\} \setminus \{P_k\}$, and after sorting, $P_k$ is in the $k$-th position. This means we need $P_k = k$.
3. If no such $k$ exists, the answer is 2 (as proven possible for $N \ge 3$).

```python
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data
    # Since we cannot use loops, we process the flat list using slicing and map
    
    # We need to group the input into test cases. 
    # Each case has N followed by N elements.
    # Because N varies, we can't use a simple reshape.
    # However, we can use a recursive-like approach or a custom reducer to group them.
    
    def group_cases(data):
        # Use a helper to partition the list into (N, P) pairs
        # Since we can't use while loops, we use a generator-like 
        # structure inside a list comprehension by indexing.
        # But N is variable, so we must be clever.
        # Actually, the most reliable way without loops is to use 
        # a reducer that accumulates the cases.
        def reducer(acc, val):
            # This is tricky because we need to know if we are reading N or P_i
            # Let's use a different approach: process the stream.
            pass

    # Given the constraints and the "no loop" rule, 
    # the most idiomatic way to handle variable length chunks 
    # is to use an iterator.
    it = iter(input_data)
    
    def get_cases():
        # Use a recursive-like structure via map/list comprehension
        # But recursion depth is an issue. 
        # Let's use a trick: since we can't use 'for' or 'while',
        # we can use 'map' with a function that pulls from the iterator.
        def process_case():
            try:
                N = int(next(it))
                P = [int(next(it)) for _ in range(N)] # Wait, range() in list comp is a loop.
                # The constraint says "no for or while loops". 
                # List comprehensions are explicitly allowed.
                # But 'for _ in range(N)' is a loop.
                # Let's use islice from itertools.
                return N, P
            except StopIteration:
                return None

    # Re-evaluating: "You must not use any for or while loops."
    # List comprehensions are allowed. "for i in range(N)" inside a 
    # list comprehension is technically a loop.
    # Let's use map() and islice() to be safe.
    from itertools import islice
    
    def solve_all():
        # Use a generator to yield cases, then map the logic over it.
        def case_generator():
            # We can't use 'while', but we can use recursion 
            # if we increase the limit, or use a clever map.
            # Actually, the simplest way to avoid 'for/while' 
            # is to use a recursive function for the T cases.
            pass

    # Let's refine the logic:
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that P[k-1] == k and 
    #       sorted(P[:k-1]) == [1...k-1] and sorted(P[k:]) == [k+1...N]
    #       This is equivalent to: P[k-1] == k and 
    #       max(P[:k-1]) < k and min(P[k:]) > k.
    # 2 ops: Otherwise.

    # To avoid all loops, we use map, filter, and list comprehensions.
    # Note: The prompt says "no for or while loops", but "list comprehensions 
    # are allowed". Usually, this means the 'for' keyword is banned 
    # EXCEPT inside comprehensions.
    
    # To handle the variable N, we can't easily use map() because 
    # we don't know the chunk sizes. 
    # But we can use a recursive function to process T cases.
    sys.setrecursionlimit(200005)
    
    def run():
        # Using a list to store the flat input
        data = input_data[1:]
        
        def get_result(n, p):
            # Check if sorted
            # We can use all() with a generator expression
            if all(p[i] == i + 1 for i in range(n)):
                return 0
            
            # Check if 1 operation suffices
            # We need k such that P[k-1] == k and 
            # max(P[0...k-2]) < k and min(P[k...n-1]) > k
            # We can precompute prefix max and suffix min.
            
            # Using list comprehensions to build prefix/suffix arrays
            # Since we can't use loops, we use a trick for prefix/suffix:
            # We can't use reduce for prefix max easily without a loop.
            # Wait, we can use itertools.accumulate!
            from itertools import accumulate
            
            pref_max = list(accumulate(p, max))
            suff_min = list(accumulate(p[::-1], min))[::-1]
            
            # k is 1-indexed. The element is p[k-1].
            # Condition: p[k-1] == k and 
            # (k==1 or pref_max[k-2] < k) and 
            # (k==n or suff_min[k] > k)
            
            # We check if any k in 1..n satisfies this.
            # We use 'any' with a generator expression.
            if any(
                p[k-1] == k and 
                (k == 1 or pref_max[k-2] < k) and 
                (k == n or suff_min[k] > k) 
                for k in range(1, n + 1)
            ):
                return 1
            
            return 2

        # To process T cases without a loop, we use a recursive function.
        def process(remaining_data):
            if not remaining_data:
                return []
            n = int(remaining_data[0])
            p = [int(x) for x in remaining_data[1:n+1]]
            return [get_result(n, p)] + process(remaining_data[n+1:])

        # However, recursion for 10^5 cases will hit depth limit.
        # Let's use a different approach to group the data.
        # We can use a list comprehension to find the boundaries of each case.
        # But that requires knowing the Ns.
        
        # Actually, the most robust way to avoid 'for' and 'while' 
        # is to use map() and a helper function.
        # But since N is variable, we can't map directly.
        # Let's use a generator and map.
        
        def case_stream():
            # We can use a generator with a yield, but that requires a loop.
            # The only way to process this is to use the 'for' in a 
            # list comprehension to iterate through the range of T.
            # But we need to track the index.
            pass

    # Let's reconsider: the constraint says "no for or while loops", 
    # but "list comprehensions are allowed".
    # I will use a list comprehension to iterate through the test cases
    # by maintaining a state in a mutable object or using a 
    # clever indexing trick.
    
    # Actually, the simplest way to handle variable N is to 
    # pre-calculate the start indices of each case.
    
    # Since I cannot use loops, I will use a recursive-like 
    # structure via a list comprehension and a helper.
    
    # Wait, if I use a list comprehension to iterate over T, 
    # I still need to slice the data.
    # Let's use a global index and a function.
    
    # Correct approach:
    # 1. Read all data.
    # 2. Use a recursive function to find the boundaries of cases.
    # 3. Use map() to apply the logic.
    # 4. Use sys.setrecursionlimit.
    
    # But there's a better way: use a generator and `itertools.islice`.
    # A generator function can use `for` loops? No, the prompt says 
    # "no for or while loops". This includes generator functions.
    
    # Let's use `functools.reduce` to partition the data.
    from functools import reduce
    
    def partition(data):
        # reduce(function, sequence, initial)
        # acc = (current_index, list_of_cases)
        def step(acc, _):
            idx, cases = acc
            n = int(data[idx])
            p = [int(x) for x in data[idx+1 : idx+1+n]]
            return (idx + 1 + n, cases + [(n, p)])
        
        # We need to call step T times.
        # We can use map(None, range(T)) and reduce.
        # But we can't use range(T) in a loop. 
        # We can use it in a list comprehension.
        return reduce(step, [None] * T, (1, []))[1]

    # Final attempt at structure:
    # Use reduce to group cases, then map the logic, then join with \n.
    
    # To avoid the O(N^2) list concatenation in reduce, 
    # use a deque or just append to a list. 
    # But append is a statement, not an expression.
    # Let's use a list and index into it.
    
    # Actually, the most efficient way to group variable length 
    # items without loops is to use a custom class and map.
    
    class State:
        def __init__(self, data):
            self.data = data
            self.ptr = 1
        def next_case(self):
            n = int(self.data[self.ptr])
            p = [int(x) for x in self.data[self.ptr + 1 : self.ptr + 1 + n]]
            self.ptr += 1 + n
            return (n, p)

    # Now we can use map(lambda _: state.next_case(), range(T))
    # This uses 'range' and 'map', and the 'for' is hidden inside map.
    
    # Let's implement the logic inside a function.
    
    from itertools import accumulate
    
    def get_ans(case):
        n, p = case
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        pref_max = list(accumulate(p, max))
        suff_min = list(accumulate(p[::-1], min))[::-1]
        
        if any(
            p[k-1] == k and 
            (k == 1 or pref_max[k-2] < k) and 
            (k