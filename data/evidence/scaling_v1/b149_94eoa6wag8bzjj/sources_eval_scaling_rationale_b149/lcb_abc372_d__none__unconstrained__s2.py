The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing data transformations over step-by-step mutations. To solve this problem without loops or recursion, I will use `functools.reduce` to maintain a stack of indices representing buildings in decreasing order of height, and list comprehensions to format the output.

```python
import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j" 
    # means Building j is a "visible" building looking right from i.
    # Specifically, j satisfies the condition if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that j is one of the indices that would 
    # remain in a monotonic decreasing stack if we processed the array from j down to i.
    # However, a simpler observation: for a fixed i, the buildings j that satisfy 
    # this are the ones that form a strictly increasing subsequence of heights 
    # starting from the first building to the right of i.
    
    # To solve this for all i efficiently without loops, we process the array from right to left.
    # We maintain a stack of indices whose heights are strictly increasing (from the perspective of the right).
    # For a building i, the number of j > i satisfying the condition is the number of 
    # elements in the stack that are "visible".
    # Actually, the condition is: j satisfies it if for all k such that i < k < j, H_k < H_j.
    # This means j is a "right-side" record.
    # For a fixed i, the valid j's are:
    # 1. j = i + 1
    # 2. The smallest j > i+1 such that H_j > H_{i+1}
    # 3. The smallest j > previous_j such that H_j > H_{previous_j}
    # and so on.
    
    # This is exactly the number of elements in a monotonic stack if we iterate from N down to 1,
    # but the stack logic is tricky without loops. 
    # Let's use the property: c_i = 1 + c_{next_greater_element(i+1)}
    # where next_greater_element(k) is the index of the first building taller than H_k to the right.
    
    # Step 1: Find the next greater element (NGE) for all indices.
    # We can use reduce to simulate the stack-based NGE algorithm.
    def find_nge(acc, idx):
        stack, nge = acc
        # Pop from stack while current height is greater than stack top height
        while stack and h[stack[-1]] < h[idx]:
            top = stack.pop()
            nge[top] = idx
        stack.append(idx)
        return (stack, nge)

    # Since I cannot use 'while' loops, I must use a different approach for NGE.
    # Wait, the constraint says "no loops". 'while' is a loop.
    # I will use a recursive-like structure via reduce or map, but recursion is banned.
    # Let's use the property that we can find NGE using a divide and conquer approach 
    # implemented via map/reduce or by using a specific functional pattern.
    
    # Actually, the most reliable way to avoid loops/recursion for NGE is to 
    # use a Segment Tree or Fenwick Tree, but those usually require loops.
    # Let's use the fact that we can find the NGE by sorting indices by height.
    
    # Correct approach without loops:
    # For each i, we want to count j > i such that H_j > max(H_{i+1} ... H_{j-1}).
    # This is equivalent to: j is a valid index if it's the first index to the right 
    # of i with height > H_{i+1}, or the first index to the right of that one with 
    # height > its height, etc.
    
    # Let's use a different observation: 
    # The number of such j's for index i is simply the number of elements 
    # in the monotonic stack (decreasing) when processing from i+1 to N.
    # But we need this for all i.
    
    # Let's use the property: c_i = 1 + c_{NGE(i+1)} if i < N-1 else 0.
    # To find NGE without loops:
    # We can use a Segment Tree implemented with list comprehensions and map.
    # Or, we can use the fact that N is 2*10^5, so we need O(N log N).
    
    # Since I cannot use loops, I will use a technique to simulate the NGE 
    # using a sorted list of heights and a Disjoint Set Union (DSU) 
    # implemented via a dictionary and a functional update pattern.
    # But DSU also needs loops for path compression.
    
    # Final attempt at a loop-free strategy:
    # Use the property that the answer for i is the number of elements 
    # in the upper hull of the points (j, H_j) for j > i.
    # Actually, the simplest loop-free way to implement NGE is to use 
    # a Divide and Conquer approach using a helper function and 
    # mapping it over ranges, but that requires recursion.
    
    # Wait, the only way to avoid loops and recursion is to use 
    # built-ins like map, filter, reduce, and list comprehensions.
    # I can implement NGE by processing heights in increasing order 
    # and using a Fenwick tree or Segment tree, but updating them 
    # requires loops.
    
    # Let's use the "Next Greater Element" logic with a stack 
    # implemented inside a reduce, but the "while" loop is the problem.
    # I can replace the "while" loop with a recursive function, 
    # but recursion is forbidden.
    
    # There is one trick: use a list as a stack and 
    # use a list comprehension to "filter" the stack.
    # But you can't mutate the stack in a way that removes multiple elements 
    # without a loop.
    
    # Let's reconsider: c_i is the number of j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements in the monotonic stack 
    # when iterating from i+1 to N.
    # This is also equal to the number of indices j > i such that 
    # H_j is a prefix maximum of the sequence H_{i+1}, ..., H_N.
    
    # For a fixed i, the answer is the number of prefix maximums of H[i+1:].
    # This is a known problem that can be solved by a Segment Tree.
    # A Segment Tree can be implemented without loops using 
    # list comprehensions for the tree structure and 
    # map/reduce for queries.
    
    # To avoid loops and recursion, I will use a Segment Tree 
    # where each node stores the maximum height in its range 
    # and the number of prefix maximums.
    
    def build_tree(l, r, heights):
        if l == r:
            return (heights[l], 1)
        mid = (l + r) // 2
        left = build_tree(l, mid, heights)
        right = build_tree(mid + 1, r, heights)
        # This is recursion. Forbidden.
        pass

    # If loops and recursion are forbidden, the only way to 
    # process the array is via reduce/map.
    # Let's use the property: c_i = 1 + c_{NGE(i+1)}
    # To find NGE without loops/recursion:
    # We can use the fact that NGE(i) is the index j > i such that 
    # H_j is the first element > H_i.
    # We can find this by processing elements in decreasing order of height.
    # We use a sorted list of indices and a way to find the next 
    # available index.
    
    # Actually, the most idiomatic "no-loop" way to solve this 
    # is to use a Segment Tree and a specific query 
    # that counts prefix maximums, implemented iteratively 
    # using list comprehensions to simulate the tree.
    # But "iterative" usually implies loops.
    
    # Let's use the property: the answer for i is the number of 
    # elements in the monotonic stack.
    # I will use a trick: use a very large number of 
    # map/reduce calls to simulate the process.
    # No, that's not possible.
    
    # Wait, the constraint to avoid loops is extremely strict.
    # Let's use the NGE property with a functional approach.
    # We can find NGEs by sorting indices by height and 
    # using a Fenwick tree to find the next index.
    # But Fenwick tree updates need loops.
    
    # Let's use the only remaining option: 
    # The problem can be solved by a Segment Tree. 
    # I can implement the Segment Tree using a flat list 
    # and use list comprehensions to perform the 
    # range-based updates/queries.
    
    # Actually, the simplest way to implement this 
    # without loops/recursion is to use 
    # `functools.reduce` to maintain the monotonic stack, 
    # and inside the reduce, use a list slice 
    # to remove elements from the stack.
    # Since I can't use `while`, I can use 
    # `bisect_right` on a sorted version of the stack 
    # if the stack is maintained sorted.
    # But the stack is sorted by height, not by index.
    
    # If I maintain a stack of (height, index) that is 
    # strictly decreasing, I can use `bisect` to find 
    # how many elements to pop.
    
    import bisect

    def step(state, h_val):
        stack, results = state
        # stack stores heights of buildings in decreasing order
        # We need to remove all heights < h_val
        # Since the stack is decreasing, we find the first 
        # element <= h_val using bisect.
        # To use bisect, we store heights negated to make it increasing.
        # stack: [-10, -8, -5, -2] (original heights 10, 8, 5, 2)
        # h_val = 6. We want to remove -5 and -2.
        # bisect_right([-10, -8, -5, -2], -6) -> index 2.
        # New stack: [-10, -8, 6]
        
        # However, the stack needs to be maintained for the 
        # "number of visible buildings" logic.
        # For a fixed i, the answer is the number of prefix maximums of H[i+1:].
        # This is solved by processing from right to left.
        # When moving from i+1 to i, we add H_{i+1} to the set.
        # The number of prefix maximums for i is 1 + (number of prefix maximums 
        # of H[i+2:] that are greater than H_{i+1}).
        pass

    # Let's use the property: 
    # The answer for i is the number of elements in the 
    # monotonic stack of H[i+1...N].
    # When we prepend H_i to the sequence, the new 
    # monotonic stack is H_i followed by all elements 
    # in the previous stack that are greater than H_i.
    
    # This is perfect! 
    # 1. Process heights from right to left.
    # 2. Maintain a monotonic stack of heights (strictly increasing).
    # 3. For height H_i, the answer is 1 + (number of elements in 
    #    the stack > H_i).
    # 4. The new stack is [H_i] + [elements in stack > H_i].
    
    # Since the stack is always sorted, we can use bisect 
    # to find the number of elements > H_i and to slice the stack.
    
    def process(state, h_val):
        stack, res = state
        # stack is sorted increasing (since we process right to left)
        # We need elements in stack > h_val.
        idx = bisect.bisect_right(stack, h_val)
        count = len(stack) - idx
        # New stack is h_val followed by elements > h_val.
        # To keep it sorted for bisect