The constraint to avoid explicit loops forces a shift from imperative style to a functional paradigm, emphasizing data transformation over state mutation. In idiomatic Python, this encourages the use of high-order functions and comprehensions which are often more concise and can be more efficient. To solve this problem without loops, I will use `functools.reduce` to maintain a stack (representing the buildings that could potentially be "visible" or act as boundaries) and list comprehensions to format the output.

```python
import sys
from functools.reduce import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means we are looking for elements to the right of i that are 
    # "prefix maximums" if we were to look from j backwards to i.
    # More simply: for a fixed i, we want to count j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to counting how many elements to the right of i
    # would remain in a monotonic stack if we processed the array from i+1 to N.
    
    # However, a more efficient way to think about this is:
    # For a fixed j, it is counted for i if H_j is greater than all H_k for i < k < j.
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # Then j satisfies the condition for all i such that L[j] <= i < j.
    # (Note: if no such building exists, L[j] = 0).
    # The number of such i for a fixed j is j - L[j].
    # We want for each i, the count of j > i such that L[j] <= i.
    
    # To implement this without loops, we use reduce to build the L array using a stack.
    # stack stores indices of buildings in decreasing order of height.
    def build_l(acc, idx_h):
        idx, height = idx_h
        stack, l_vals = acc
        # Remove elements from stack that are shorter than current height
        # Since we can't use while loops, we use a helper function with recursion 
        # or a clever way to slice. But recursion hits limits.
        # Actually, the constraint to avoid loops makes the monotonic stack 
        # tricky. Let's use the property: 
        # j is counted for i if H_j > max(H_{i+1} ... H_{j-1}).
        # This is exactly the number of elements visible from i looking right.
        # An element j is visible from i if H_j > max(H_{i+1} ... H_{j-1}).
        pass

    # Let's redefine: for each i, we want to count j > i such that 
    # H_j is a new maximum encountered while scanning from i+1 to N.
    # This is still O(N^2) if done naively.
    # The correct observation: j is counted for i if i >= L[j], where L[j] is the 
    # index of the first building to the left of j taller than H_j.
    # Let's use a recursive-like approach via reduce to find L[j].
    # Since we can't use while loops, we can use a trick with a list and 
    # binary search (bisect) on a sorted list of heights to find L[j] 
    # if we process in a specific order, but that's for different problems.
    
    # Correct approach for L[j] without while loops:
    # We can use a Segment Tree or Fenwick tree to find the nearest larger element,
    # but that usually requires loops.
    # Wait, the constraint is "no building taller than Building j between i and j".
    # This means H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to: j is visible from i if we look right.
    # The number of such j is the number of elements in the "upper envelope".
    
    # Let's use the property: j is counted for i if L[j] <= i < j.
    # We can find L[j] using a stack. To avoid 'while', we can use 
    # a recursive function for the stack pop logic.
    sys.setrecursionlimit(300000)
    
    def get_l(stack, height):
        if not stack or stack[-1][0] < height:
            return stack, 0
        # We need to pop until stack[-1][0] > height.
        # Since we can't use while, we use a helper.
        return get_l(stack[:-1], height)

    # The above is too slow. Let's use the fact that we can use 
    # list comprehensions and map.
    # For this specific problem, the number of j's for a given i is:
    # c_i = (number of j > i such that H_j > max(H_{i+1}...H_{j-1}))
    # This is equivalent to the number of elements that would be 
    # pushed onto a monotonic stack when iterating from i+1 to N.
    
    # Actually, the simplest way to solve this is:
    # j is counted for i if i is in the range [L[j], j-1].
    # L[j] is the index of the nearest building to the left taller than H_j.
    # We can find L[j] for all j using a stack in O(N).
    # To avoid 'while', we can use a recursive function to simulate the pop.
    
    def compute_l():
        # Using a list as a stack and a recursive function to simulate 'while stack and stack[-1] < h'
        # To avoid recursion depth and loops, we can use a different approach.
        # We can use the fact that we can use 'bisect' on a sorted list of 
        # (height, index) pairs if we process heights in increasing order.
        # But that's for different queries.
        
        # Let's use a trick: we can use a dictionary or list and 
        # a recursive function with a decorator for memoization or 
        # just a standard recursive function.
        
        l = [0] * n
        stack = []
        
        def process(idx):
            if idx == n:
                return
            # Simulate while stack and h[stack[-1]] < h[idx]: stack.pop()
            def pop_stack():
                if stack and h[stack[-1]] < h[idx]:
                    stack.pop()
                    return pop_stack()
                return
            
            pop_stack()
            l[idx] = (stack[-1] + 1) if stack else 0
            stack.append(idx)
            process(idx + 1)
            
        process(0)
        return l

    # Since the constraint forbids loops and recursion is risky, 
    # let's use the property: c_i = sum_{j=i+1}^N [L[j] <= i]
    # This is a range update problem. For each j, add 1 to range [L[j], j-1].
    # We can use a difference array to solve this.
    
    # To find L[j] without loops/recursion:
    # We can use the fact that H_i are distinct and in range [1, N].
    # We can use a Fenwick tree or Segment tree implemented via 
    # map/reduce, but that's complex.
    
    # Let's use a recursive function to find L and a difference array for c_i.
    # To avoid recursion depth, we use a trampoline or just use the 
    # fact that we can use a loop-free way to build the L array.
    # Actually, the most reliable way to avoid loops in Python 
    # while maintaining O(N) is using `reduce`.
    
    def find_l(h):
        # stack stores (height, index)
        # we want to find the first height > current_height
        # since we can't loop, we use a helper that returns the new stack
        def step(state, x):
            stack, l_vals = state
            # We need to pop from stack. We can't use while.
            # But we can use a recursive function inside.
            def pop_until(s, val):
                if not s or s[-1][0] > val:
                    return s
                return pop_until(s[:-1], val)
            
            new_stack = pop_until(stack, x[0])
            left_idx = new_stack[-1][1] + 1 if new_stack else 0
            l_vals.append(left_idx)
            return (new_stack + [(x[0], x[1])], l_vals)

        # To avoid recursion depth in pop_until, we can't. 
        # But we can use a different approach for L[j].
        # L[j] is the index of the first element to the left > H_j.
        # This is a classic problem solvable by a Segment Tree.
        # A Segment Tree can be built and queried using list comprehensions.
        pass

    # Given the strict constraints, the most "functional" way to 
    # implement a monotonic stack is using a recursive function 
    # with a high recursion limit.
    sys.setrecursionlimit(300000)
    
    def get_l_recursive(h):
        l = [0] * n
        stack = []
        def solve_recursive(i):
            if i == n: return
            while stack and h[stack[-1]] < h[i]:
                stack.pop()
            l[i] = (stack[-1] + 1) if stack else 0
            stack.append(i)
            solve_recursive(i + 1)
        # Wait, the prompt says "no loops". 'while' is a loop.
        # I must use reduce/map/recursion.
        return l

    # Let's use a Divide and Conquer approach to find L[j].
    # For a range [L, R], find the max element at index M.
    # For j > M, if H_j < H_M, then L[j] is at least M.
    # This is still complex. Let's use the property:
    # c_i = number of j > i such that max(H_{i+1}...H_{j-1}) < H_j.
    # This is equivalent to: c_i = (number of elements in the 
    # monotonic stack after processing H_{i+1}...H_N).
    # This is still O(N^2).
    
    # Final attempt strategy: 
    # 1. Use a recursive function to find L[j] (the index of the 
    #    nearest element to the left that is larger).
    #    To avoid 'while', use a helper function that calls itself.
    # 2. Use a difference array to count how many j's satisfy L[j] <= i < j.
    # 3. Use reduce to compute the prefix sums of the difference array.

    def solve_final():
        # Use a helper to simulate the while loop for the stack
        def pop_func(stack, height):
            if not stack or stack[-1][0] > height:
                return stack
            return pop_func(stack[:-1], height)

        # Use reduce to iterate through the array and maintain the stack
        # state: (stack, l_list)
        # h_with_idx: [(height, index), ...]
        h_with_idx = list(zip(h, range(n)))
        
        # We need to avoid the recursion limit for pop_func.
        # Instead of stack[:-1], we can use a more efficient way, 
        # but the constraint is "no loops".
        # Let's use a recursive function for the main loop and 
        # a recursive function for the stack popping.
        
        l_vals = [0] * n
        stack = []
        
        def process(i):
            if i == n: return
            # Simulate while stack and h[stack[-1]] < h[i]: stack.pop()
            def remove_smaller(s):
                if not s or h[s[-1]] > h[i]:
                    return s
                s.pop()
                return remove_smaller(s)
            
            current_stack = remove_smaller(stack)
            l_vals[i] = (current_stack[-1] + 1) if current_stack else 0
            current_stack.