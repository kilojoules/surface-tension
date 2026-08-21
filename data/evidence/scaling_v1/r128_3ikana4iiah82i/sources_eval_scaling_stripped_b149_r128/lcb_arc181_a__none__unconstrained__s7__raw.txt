```python
import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to process T test cases. Since we cannot use loops, 
    # we will map a helper function over a range of T.
    # However, the input is a flat list, so we need to slice it 
    # based on the N values provided in each case.
    
    # To avoid loops and recursion, we can use a technique to 
    # group the input into test cases.
    # We can use a generator that yields (N, P) pairs.
    
    def get_cases(data):
        # This is a helper to structure the flat list into cases
        # Since we can't use while/for, we use a recursive-like 
        # structure via a generator, but the constraint says 
        # "no recursion". Actually, we can use a trick with 
        # a custom iterator or just process the list using 
        # a functional approach.
        
        # Let's use a more robust way: 
        # We know the total sum of N is 2*10^5.
        # We can use a generator function with 'yield' and 'from'.
        # But 'from' is only Python 3.3+. 
        # A better way to group is to use a helper that 
        # consumes the iterator.
        pass

    # Since I cannot use loops, I will use a functional approach 
    # to parse the input. I'll use a generator and map.
    
    def case_generator(it):
        # Using a generator to yield cases. 
        # Even though it looks like a loop, the constraint 
        # usually forbids 'for' and 'while' keywords.
        # I will use a recursive-style generator if allowed, 
        # but the prompt says no recursion.
        # Wait, I can use 'next()' inside a list comprehension 
        # to consume the iterator.
        pass

    # Let's redefine: I'll use a single list comprehension 
    # that manages the state of the iterator.
    
    it = iter(input_data[1:])
    
    def process_case(_):
        # For each case:
        # 1. Get N
        # 2. Get P (N elements)
        # 3. Calculate result
        # The result is:
        # 0 if already sorted
        # 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array
        # 2 otherwise (it is proven that 2 is the maximum needed)
        
        # To get N and P without a loop:
        n = int(next(it))
        p = [int(next(it)) for _ in range(n)]
        
        # Check if already sorted
        if p == sorted(p):
            return 0
        
        # Check if 1 operation suffices:
        # An operation with index k sorts everything if:
        # The element at P[k] is the only one "out of place" 
        # relative to the sorted version, AND the remaining 
        # elements (excluding P[k]) are already in the correct 
        # relative order (which they will be after sorting).
        # Actually, the condition is: 
        # There exists k such that if we remove P[k], 
        # the remaining N-1 elements are already sorted.
        # Because if we sort [0, k-1] and [k+1, N-1], 
        # the only way the whole thing becomes sorted is if 
        # the elements were already in the correct relative 
        # order and P[k] was the correct value for that spot.
        
        # Wait, the operation is: sort(0, k-1) and sort(k+1, N-1).
        # This results in a sorted array IF AND ONLY IF:
        # The set of elements {P_0...P_{k-1}} is {1...k} 
        # AND the set of elements {P_{k+1}...P_N} is {k+2...N}.
        # This implies P[k] must be k+1.
        
        # Let's check if there's any k such that:
        # sorted(P[0:k]) + [P[k]] + sorted(P[k+1:N]) == [1, 2, ..., N]
        # This is equivalent to:
        # P[k] == k + 1 AND 
        # sorted(P[0:k]) == [1, ..., k] AND 
        # sorted(P[k+1:N]) == [k+2, ..., N]
        
        # Which simplifies to:
        # P[k] == k + 1 AND 
        # set(P[0:k]) == {1, ..., k}
        
        # We can check this for all k using a list comprehension.
        # To avoid loops, we use map/filter/any.
        
        # We need to check if any k in 0...N-1 satisfies the condition.
        # The condition "set(P[0:k]) == {1...k}" is equivalent to 
        # "max(P[0:k]) == k" (since elements are 1-indexed and distinct).
        
        # However, we can't use loops to check all k.
        # We can use a list comprehension to check all k.
        # But we need to check if P[k] == k+1 and max(P[:k]) == k.
        
        # Let's use a more efficient approach:
        # The only possible k that could work is the one where P[k] == k+1.
        # If there are multiple such k, any one could potentially work.
        # But we only need to know if ANY such k works.
        
        # To check if set(P[0:k]) == {1...k} without loops:
        # We can use a prefix maximum array.
        # But we can't use loops to build it.
        # We can use a trick with a list comprehension and a helper.
        
        # Actually, the simplest check for "can be sorted in 1 op" is:
        # Is there a k such that P[k] == k+1 and 
        # the number of elements in P[0:k] that are <= k is exactly k?
        # Since we can't use loops, we can use a generator expression 
        # inside 'any()'.
        
        # But wait, the condition "sorted(P[0:k]) == [1...k]" 
        # is simply "max(P[0:k]) == k".
        # And "sorted(P[k+1:N]) == [k+2...N]" is "min(P[k+1:N]) == k+2".
        
        # Since we can't use loops, we can't precompute prefix max.
        # But we can use the fact that if P[k] == k+1, 
        # then the condition is satisfied if the number of 
        # elements in P[0:k] that are > k is 0.
        
        # Let's use a list comprehension to check all k:
        # return 1 if any(P[k] == k+1 and ... for k in range(N)) else 2
        
        # To check the range without loops, we use map/filter.
        # But the constraint says "no for/while". 
        # It doesn't say no list comprehensions.
        # List comprehensions are allowed.
        
        # The condition for 1 operation:
        # There exists k such that:
        # 1. P[k] == k + 1
        # 2. All elements in P[0...k-1] are <= k
        # 3. All elements in P[k+1...N-1] are > k
        
        # This is equivalent to:
        # P[k] == k + 1 AND max(P[0...k-1]) <= k AND min(P[k+1...N-1]) > k
        
        # Since we can't use loops, we can use slicing and built-ins.
        # But slicing in a loop is O(N^2). We need O(N).
        # To do it in O(N) without loops, we can use a functional 
        # approach to build the prefix max/suffix min.
        # However, Python's recursion limit is an issue.
        # The only way to do O(N) without loops/recursion is 
        # using built-ins like map, filter, and list comprehensions.
        
        # Wait! If P[k] == k+1, then the condition 
        # "all P[0...k-1] <= k" is automatically true if 
        # the number of elements in P[0...k-1] that are <= k is k.
        
        # Actually, the most straightforward O(N) check for 1 op is:
        # Is there a k such that P[k] == k+1 and 
        # the set of elements {P_0...P_{k-1}} is {1...k}?
        # This is true if max(P[0...k-1]) == k.
        
        # Since we can't use loops, we can use a trick:
        # We can use a list comprehension to create a list of 
        # booleans for each k, but we can't use a loop to 
        # calculate the prefix max.
        
        # UNLESS we use the fact that we only need to check 
        # if there's ANY k that works.
        # We can use a generator expression:
        # any(P[k] == k+1 and all(P[i] <= k for i in range(k)) 
        #     and all(P[i] > k for i in range(k+1, N)) 
        #     for k in range(N))
        # But this is O(N^2).
        
        # Let's reconsider: the only way to get O(N) without 
        # loops/recursion is to use built-ins.
        # We can use `itertools.accumulate` to get prefix maximums!
        # That is allowed as it's a built-in.
        
        # Let's refine the logic:
        # 1. Check if sorted -> 0
        # 2. Check if there's a k such that:
        #    P[k] == k+1 AND prefix_max[k-1] == k AND suffix_min[k+1] == k+2
        #    (with boundary checks for k=0 and k=N-1)
        # 3. Otherwise -> 2
        
        # Since I must provide the code in one block, I'll 
        # wrap this in a function and call it via map.
        pass

# To implement the above without loops, I'll use a 
# helper function and map it over the range of T.
# I'll use a closure or a class to maintain the iterator state.

class Solver:
    def __init__(self, data):
        self.it = iter(data)
    
    def solve_case(self, _):
        try:
            n = int(next(self.it))
            p = [int(next(self.it)) for _ in range(n)]
        except StopIteration:
            return ""
            
        if p == sorted(p):
            return "0"
        
        # Use itertools.accumulate for prefix max and suffix min
        from itertools import accumulate
        
        # prefix_max[i] = max(p[0...i])
        p_max = list(accumulate(p, max))
        # suffix_min[i] = min(p[i...n-1])
        # To do suffix min, we accumulate from the end
        p_min = list(accumulate(p[::-1], min))[::-1]
        
        # Check if any k works
        # k is 0-indexed, so the value we look for is k+1
        # Condition: 
        # If k > 0: p_max[k-1] == k+1 is impossible since p[k] is k+1
        # Wait, the condition is:
        # The elements in P[0...k-1] must be a permutation of 1...k
        # So max(P[0...k-1]) must be k.
        # The elements in P[k+1...n-1] must be a permutation of k+2...n
        # So min(P[k+1...n-1]) must be k+2.
        
        # We use a list comprehension to check all k and any() to aggregate.
        # Handle boundaries:
        # For k