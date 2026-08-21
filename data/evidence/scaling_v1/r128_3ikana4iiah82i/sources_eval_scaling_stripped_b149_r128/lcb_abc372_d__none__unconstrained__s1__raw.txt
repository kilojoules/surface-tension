```python
import sys
from bisect import bisect_right

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] <= H[j] for all i < k < j.
    # This is equivalent to saying that H[j] must be a "running maximum" 
    # of the sequence H[i+1...j].
    # However, the problem asks for the count for each i.
    # Let's rephrase: for a fixed j, how many i < j satisfy the condition?
    # The condition is: max(H[i+1...j-1]) <= H[j].
    # This means i must be such that all buildings between i and j are shorter than H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = 0 (using 1-indexing).
    # Then any i from L[j] to j-1 satisfies the condition.
    # The number of such i is j - L[j].
    # But we need the count for each i. 
    # Let's use the property: j satisfies the condition for i if 
    # H[j] is the maximum of the range [i+1, j].
    # This is a classic problem that can be solved by iterating backwards.
    # For a fixed i, the valid j's are the indices of the 
    # "upper envelope" of the heights to the right.
    # Specifically, the first j > i is always valid.
    # The second valid j is the first index k > j such that H[k] > H[j], and so on.
    # Wait, the condition is: no building BETWEEN i and j is taller than H[j].
    # This means H[j] >= max(H[i+1], ..., H[j-1]).
    # This is exactly the definition of a record-breaking value from the left
    # if we start the sequence at i+1.
    
    # Let's use the observation: 
    # For a fixed i, the valid j's are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1] is NOT required.
    # Actually, the condition is: H[j] >= max(H[i+1...j-1]).
    # This means the sequence of H[j] for valid j's is non-decreasing.
    # The valid j's are the indices where the prefix maximum of H[i+1...N] increases.
    # Since all H_i are distinct, it's strictly increasing.
    
    # To solve this for all i efficiently:
    # We are looking for the number of j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting how many elements to the right of i 
    # are greater than all elements between them and i.
    
    # Let's use a Segment Tree or Fenwick Tree? No, the range depends on i.
    # Let's use the property: j is valid for i if H[j] is the maximum of H[i+1...j].
    # This is a known problem. The number of such j's for a fixed i is the 
    # number of elements in the "right-side" monotonic stack if we processed 
    # the array from i+1 to N.
    
    # Correct approach:
    # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting how many j > i are "visible" from i.
    # A building j is visible from i if it's the tallest building in the range [i+1, j].
    # This is the number of elements in the sequence H[i+1...N] that are 
    # strictly greater than all preceding elements in that sequence.
    
    # We can solve this by processing the array from right to left.
    # For a fixed i, the valid j's are:
    # The first element H[i+1], then the first element to the right of that 
    # which is taller, and so on.
    # This looks like a jump pointer problem (Binary Lifting).
    # Let next[j] be the index of the first building to the right of j that is taller than H[j].
    # The valid j's for i are: (i+1), next[i+1], next[next[i+1]], ...
    # The number of such j's is the depth of the node (i+1) in the forest 
    # defined by the 'next' pointers.
    
    # 1. Find 'next' array using a stack
    # next[j] = index of first k > j such that H[k] > H[j]
    # We use a stack to find the next greater element.
    
    # Since we cannot use loops, we use a technique to simulate the stack 
    # or use a recursive-like structure. However, Python's recursion limit 
    # is an issue. We can use a list comprehension with a helper function 
    # if we are careful, but the 'next' array is standard.
    # To avoid loops, we can use the fact that we can find the next greater 
    # element using a divide and conquer approach or by using a 
    # library-like implementation.
    
    # Actually, the simplest way to implement the 'next' array without 
    # explicit loops is to use a recursive function with a decorator 
    # for memoization, but that's still recursion.
    # Let's use the property that we can find the next greater element 
    # by processing the array in a specific order.
    
    # Wait, the constraint allows list comprehensions. 
    # We can find the next greater element for all indices using 
    # a technique involving sorting and a Fenwick tree, but that's complex.
    # Let's use the "stack" via a reduction or a clever trick.
    # Actually, we can use a recursive function to find the next greater 
    # element and increase the recursion limit.
    
    sys.setrecursionlimit(300000)
    
    # To find the next greater element without loops:
    # We can use a divide and conquer approach.
    def find_next(l, r):
        if l == r:
            return [N] # Using N as the boundary (0-indexed, so N is out of bounds)
        mid = (l + r) // 2
        left_next = find_next(l, mid)
        right_next = find_next(mid + 1, r)
        
        # Merge step: for each element in the left half, its next greater 
        # might be in the right half.
        # This is getting complex. Let's use the stack-based approach 
        # inside a function and call it. The constraint says "no loops", 
        # but usually, this means no 'for' or 'while'. 
        # We can use 'map', 'filter', and 'reduce'.
        
        return [] # Placeholder

    # Let's use the 'next' array logic with a functional approach.
    # To find the next greater element for all i:
    # We can use a recursive function that processes the array.
    # But since I must avoid loops, I'll use a trick with `reduce`.
    
    from functools import reduce
    
    # The state for reduce will be (stack, results_list)
    # We process the array from right to left.
    def step(state, x):
        stack, res = state
        # Remove elements from stack that are smaller than current x
        # Since we can't use a while loop, we use a recursive filter or 
        # a clever slice. But the stack is sorted.
        # We can use bisect_right to find the first element taller than x.
        # However, the stack contains heights, and we need the indices.
        pass

    # Let's reconsider: the number of j's for i is 1 + count(next[i+1]).
    # This is a tree structure. The answer for i is depth(i+1).
    # We can find the 'next' array using a standard stack algorithm 
    # implemented via reduce.
    
    def get_next_array(heights):
        # We process indices from N-1 down to 0.
        # state: (stack_of_indices, next_array)
        def reducer(state, i):
            stack, next_arr = state
            # We need to pop from stack while H[stack[-1]] < H[i]
            # To do this without a loop, we can use a recursive function.
            def pop_smaller(s):
                if not s or heights[s[-1]] > heights[i]:
                    return s
                return pop_smaller(s[:-1])
            
            new_stack = pop_smaller(stack)
            val = new_stack[-1] if new_stack else N
            return (new_stack + [i], [val] + next_arr)
        
        # Using range() in reduce is allowed as it's the iterator.
        # We process indices in reverse.
        final_state = reduce(reducer, range(N-1, -1, -1), ([], []))
        return final_state[1]

    # Now we have the next_array where next_arr[i] is the index of the 
    # first building taller than H[i] to its right.
    # The answer for i is the number of jumps to reach N.
    # We can use binary lifting to find the distance to N.
    # But the number of jumps is just the depth in the forest.
    # We can calculate this using another reduce or recursion.
    
    # Let's refine the whole process:
    # 1. Get next_array using the reduce/pop_smaller method.
    # 2. Calculate depths using a dictionary and recursion with memoization.
    
    # Since I cannot use loops, I will wrap the logic in a function.
    # The 'pop_smaller' is recursive, so I'll increase the limit.
    
    # Final structure:
    # heights = H
    # next_arr = get_next_array(heights)
    # depth_map = {}
    # def get_depth(i):
    #     if i == N: return 0
    #     if i in depth_map: return depth_map[i]
    #     depth_map[i] = 1 + get_depth(next_arr[i])
    #     return depth_map[i]
    # results = [get_depth(i+1) for i in range(N)]
    # Note: for i=N, the answer is 0.
    
    # Let's implement this.
    
    # Re-defining the logic inside a single execution block.
    # I'll use a helper function for the recursion.
    
    def run():
        # Using the logic discussed
        # We need to handle the indices carefully.
        # H is 0-indexed. Building 1 is H[0].
        # For i=1 (index 0), we look at j > 0.
        # The first candidate is j=1 (index 1).
        # Then next[1], next[next[1]]...
        
        # To avoid the 'pop_smaller' recursion limit on sorted arrays,
        # we can use a different approach for next_greater_element.
        # But with setrecursionlimit, it should pass.
        
        # Actually, the most robust way to find next greater element 
        # without loops is using a Segment Tree or Merge Sort approach.
        # But that's overkill. Let's use the reduce + recursion.
        
        # Wait, the 'pop_smaller' is called for each i. 
        # The total number of pops across all i is N.
        # The recursion depth of 'pop_smaller' is the number of elements 
        # popped. In the worst case (sorted array), it could be N.
        # So sys.setrecursionlimit(300000) is necessary.
        
        # Let's put it all together.
        pass

# Since I must provide the code in one block, I will integrate the 
# logic into a single function call.

import sys
from functools import reduce

# Increase recursion depth for the 'pop_smaller' and 'get_depth' functions
sys.setrecursionlimit(300000)

def solve():
    input_data = sys