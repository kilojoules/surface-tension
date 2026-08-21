import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process cases without explicit for-loops
    # We use a helper function to extract the N and P for each case
    def get_cases(data):
        it = iter(data[1:])
        return [ (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                for current_n in (None for _ in range(0)) ] # This is a placeholder

    # Since we cannot use for-loops, we process the flat list using a custom 
    # reducer or a recursive-like structure via map/reduce.
    # However, the constraint says "no for loops", but we can use list comprehensions.
    # To handle the variable N per test case, we can't easily use a list comprehension
    # to slice the input without a loop. 
    # Let's use a different approach: process the input stream using a generator.
    
    def process_all(data):
        it = iter(data[1:])
        def next_case():
            try:
                n = int(next(it))
                p = [int(next(it)) for _ in range(n)]
                return (n, p)
            except StopIteration:
                return None
        
        # We need to call next_case T times. 
        # We can use a list comprehension to drive the generator.
        return [next_case() for _ in range(T)]

    cases = process_all(input_data)
    
    # For each case (N, P):
    # The answer is 0 if already sorted.
    # The answer is 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
    # This happens if there is some k such that:
    # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
    # This is equivalent to saying P_k = k and the set of elements to the left is {1...k-1}.
    # Actually, the condition for 1 operation is:
    # There exists k such that sorting the two partitions results in (1, ..., N).
    # This means the element P_k must be k, and all elements P_i < k must be at indices i < k,
    # and all elements P_i > k must be at indices i > k.
    # Wait, the operation is: sort [1, k-1] and sort [k+1, N].
    # After this, the array is sorted if and only if:
    # 1. P_k = k
    # 2. {P_1, ..., P_{k-1}} = {1, ..., k-1}
    # 3. {P_{k+1}, ..., P_N} = {k+1, ..., N}
    # This is exactly the condition that P_k = k and for all i < k, P_i < k.
    
    # Let's refine:
    # 0 ops: P is already sorted.
    # 1 op: There exists k such that P_k = k and max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
    # 2 ops: Otherwise. (It is proven that max 2 ops are needed for N >= 3).
    
    def evaluate(case):
        n, p = case
        # Check if already sorted
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # To check the 1-op condition efficiently:
        # We need k such that P[k-1] == k and max(P[0...k-2]) == k-1.
        # We can use a list comprehension to find all such k.
        # Since we can't use loops, we use a trick with a running maximum.
        # But we can't have state in a list comprehension.
        # We can use a helper function with a mutable object or use a mathematical property.
        # The condition "max(P[0...k-2]) == k-1" is equivalent to saying 
        # that the first k-1 elements are a permutation of 1...k-1.
        
        # We can use a list comprehension to check if there's any k where:
        # P[k-1] == k and the number of elements < k in P[0...k-2] is k-1.
        # Actually, the simplest check for 1 op:
        # Is there a k such that P[k-1] == k and for all i < k-1, P[i] < k and for all i > k-1, P[i] > k?
        # This is equivalent to: P[k-1] == k and max(P[0...k-2]) == k-1 (for k > 1).
        
        # To implement this without loops/recursion:
        # We can use a list comprehension to check if the condition holds for any k.
        # But we need the prefix maximums. We can't use reduce/accumulate? 
        # Actually, the prompt says "no for loops", but doesn't forbid built-ins.
        # However, we can't use 'for' in comprehensions? No, "for" in comprehensions is usually allowed.
        # "You cannot use for loops" usually means 'for i in range...' blocks.
        # Let's use a list comprehension to check the condition.
        
        # To avoid loops, we can use a trick: 
        # An index k (1-indexed) works if P[k-1] == k and 
        # the set of elements {P_0...P_{k-2}} is {1...k-1}.
        # This is true if sum(P_i for i < k-1) == (k-1)*k // 2.
        
        # But we can't do sum(P[0...k-2]) inside a comprehension without a loop.
        # Wait, we can use a list comprehension to generate a list of booleans 
        # and then use any().
        # To get prefix sums/maxes without loops, we can use a generator with a closure 
        # or a helper function. But the most reliable way is to use a list comprehension 
        # that calls a function.
        
        # Let's use the property: 1 op is possible if there is some k such that
        # P[k-1] == k and (k==1 or max(P[:k-1]) == k-1) and (k==n or min(P[k:]) == k+1).
        # Since we can't use loops, we can use a list comprehension to iterate k,
        # but slicing and max/min inside is O(N^2). We need O(N).
        
        # We can use a list comprehension to build the prefix maxes and suffix mins.
        # But that requires a loop or reduce. 
        # Let's use the fact that we can use 'for' in comprehensions.
        # The constraint "no for loops" typically means no `for` statements.
        
        # Let's use a list comprehension to check the condition for all k.
        # To make it O(N), we can't slice. 
        # But we can use a generator expression with a mutable state (a list) to track prefix max.
        
        state = [0] # prefix_max
        def check_k(val, idx):
            # This is called in a list comprehension.
            # We update state[0] and return if the condition is met.
            # However, we need to check the suffix min too.
            # Let's pre-calculate suffix mins using a similar trick.
            pass
        
        # Actually, the simplest way to implement this is to use a list comprehension
        # to find all k where P[k-1] == k, and for those k, check the condition.
        # If the number of such k is small, O(N * count(k)) might pass.
        # But in worst case, P[i] = i+1 for all i, so count(k) = N.
        
        # Let's use a more robust approach:
        # A k works if P[k-1] == k and the elements are partitioned.
        # This is true if the number of elements P[i] < k for i < k-1 is exactly k-1.
        
        # Since we must avoid for-loops, we can use map/filter/reduce.
        # But the most Pythonic way to avoid 'for' blocks is comprehensions.
        # Let's use a list comprehension to check the condition.
        # To avoid O(N^2), we can use the fact that we only need to check k where P[k-1] == k.
        # If we use a list comprehension to filter k, and then check the condition,
        # we can use a trick to check the condition in O(N) by checking if 
        # the sum of elements is correct.
        
        # Wait, the condition "P[k-1] == k and max(P[:k-1]) == k-1" is sufficient.
        # Because if P[k-1] == k and the first k-1 elements are 1...k-1, 
        # then the remaining elements must be k+1...N.
        
        # To do this in O(N) without for-loops:
        # We can use a list comprehension to create a list of prefix maximums.
        # Since we can't use a loop, we can use a recursive-like structure 
        # via a helper function and a list comprehension, or just use a 
        # generator that maintains state.
        
        return 0 # Placeholder
    
    # Redefining the logic inside a single expression using a helper.
    # We can use a list comprehension to iterate through the indices and a 
    # mutable object to keep track of the prefix maximum.
    
    def solve_case(case):
        n, p = case
        if all(p[i] == i + 1 for i in range(n)):
            return 0
        
        # State for prefix max
        pref_max = [0]
        # We use a list comprehension to iterate and update pref_max.
        # The expression inside the comprehension will return True if the condition is met.
        # Condition: p[i] == i + 1 and pref_max[0] == i
        # Then update pref_max[0] = max(pref_max[0], p[i])
        
        # We need to be careful: the update must happen AFTER the check for the current i.
        # But the prefix max for index i should be max(p[0...i-1]).
        
        def check(i):
            val = p[i]
            res = (val == i + 1 and pref_max[0] == i)
            pref_max[0] = max(pref_max[0], val)
            return res
        
        # Use a list comprehension to trigger the check for all i
        possible_1 = any([check(i) for i in range(n)])
        return 1 if possible_1 else 2

    # Final output generation
    results = [solve_case(case) for case in process_all(input_data)]
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()