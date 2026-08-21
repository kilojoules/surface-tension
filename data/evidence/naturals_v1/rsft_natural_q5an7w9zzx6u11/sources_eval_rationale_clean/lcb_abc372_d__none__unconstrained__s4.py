The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. In this specific problem, we need to count how many buildings to the right of index $i$ are "visible" (no building between $i$ and $j$ is taller than building $j$). This is equivalent to counting how many buildings $j > i$ form a new maximum when scanning from $i+1$ to $N$. However, a more efficient way to view this is: building $j$ is counted for building $i$ if $H_j > \max(H_{i+1}, \dots, H_{j-1})$. This means building $j$ is a "right-side" visible building if it is taller than all buildings between it and the building to its left that is taller than it.

Actually, the condition "no building taller than Building $j$ between $i$ and $j$" means that for a fixed $j$, it is counted for all $i < j$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is satisfied for all $i$ from the index of the first building to the left of $j$ that is taller than $H_j$ (exclusive) up to $j-1$. 

If $L_j$ is the index of the nearest building to the left of $j$ such that $H_{L_j} > H_j$ (and $L_j = 0$ if no such building exists), then building $j$ contributes to the count $c_i$ for all $i$ in the range $[L_j, j-1]$. We can solve this using a monotonic stack to find $L_j$ and a difference array (or prefix sums) to calculate the final counts. Since loops are forbidden, I will use `map`, `reduce`, and list comprehensions.

```python
import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # To find the nearest building to the left that is taller than H_j
    # We use a monotonic stack approach. Since we can't use while/for loops,
    # we use a recursive-like structure via a fold (reduce) to maintain the stack.
    # stack stores indices of buildings in decreasing order of height.
    
    def find_left_taller(acc, idx):
        height = h[idx]
        stack = acc[0]
        # Remove elements from stack that are shorter than current height
        # We use a list comprehension to simulate the popping of the stack
        # However, a simple list comprehension doesn't allow mutation.
        # We find the first index in the stack that is taller than current height.
        # Since we need to avoid loops, we use a helper to slice the stack.
        
        # To avoid loops and recursion, we can't easily "pop" until a condition.
        # But we can use the fact that we only need the index of the first element 
        # in the stack > current height.
        # Given the constraints and the "no loop" rule, we use a trick with 
        # a functional approach to find the left boundary.
        pass

    # Re-evaluating: The "no loop" constraint is very strict. 
    # Let's use a different approach. 
    # Building j is counted for i if H_j > max(H_{i+1を...H_{j-1}}).
    # This is true if i is in range [L_j, j-1] where L_j is the index of 
    # the first building to the left of j taller than H_j.
    
    # Since we cannot use while loops, we can use a recursive function 
    # for the monotonic stack, but Python's recursion limit is an issue.
    # However, we can use a list comprehension to process the stack 
    # if we can find a way to represent the state.
    
    # Actually, the most "functional" way to implement a monotonic stack 
    # without loops is using a reduce that passes the stack along.
    # To handle the "pop" logic without a while loop, we can use 
    # a binary search on the stack if we maintain it carefully, 
    # but the stack is sorted by height, not index.
    
    # Wait, the stack is sorted by height (descending). 
    # We can use bisect_right on the negated heights of the stack.
    from bisect import bisect_right
    
    def process_stack(state, idx):
        stack_indices, stack_heights = state
        curr_h = h[idx]
        # Find first height in stack > curr_h. 
        # stack_heights is maintained in descending order.
        # To use bisect, we need ascending. Let's store -heights.
        # stack_heights: [-10, -8, -5, -2] (corresponding to heights 10, 8, 5, 2)
        # We want the first index where -height < -curr_h.
        
        # Since we can't use while loops to pop, we can't easily maintain 
        # the stack in a way that allows binary search for the 'left taller' 
        # because the indices we remove are not necessarily the ones we keep.
        # Actually, the monotonic stack for 'nearest greater' removes all 
        # elements smaller than current. The remaining stack is always sorted.
        
        # Let's use a recursive function with a decorator to handle the 
        # stack popping, but the prompt says no loops. Recursion is allowed.
        # But for N=2e5, we need sys.setrecursionlimit.
        return state

    # Given the strict "no loop" constraint, the only way to implement 
    # a monotonic stack is via recursion or a very clever reduce.
    # Let's use a recursive function to find L_j for all j.
    
    sys.setrecursionlimit(300000)
    
    def get_l_values(idx, stack, results):
        if idx == n:
            return results
        
        # Pop elements smaller than current height
        def pop_smaller(s):
            if not s or h[s[-1]] > h[idx]:
                return s
            return pop_smaller(s[:-1])
        
        new_stack = pop_smaller(stack)
        l_val = new_stack[-1] + 1 if new_stack else 0
        
        # We use a list to store results and update it
        results.append(l_val)
        return get_l_values(idx + 1, new_stack + [idx], results)

    # The above is still recursive. Let's use a different approach.
    # We can use a Segment Tree or Fenwick Tree to find the first index to the left 
    # with height > H_j. But that usually requires loops.
    
    # Let's use the property: building j is visible from i if 
    # H_j is the maximum in range [i+1, j].
    # This is a classic problem solvable by a monotonic stack.
    # To implement it without loops, I will use `functools.reduce`.
    # To handle the "while" part of the stack, I'll use a helper function 
    # that uses slicing and recursion.
    
    from functools import reduce
    
    def solve_final():
        # Use a helper to simulate the while loop for popping the stack
        def pop_until(stack, height):
            if not stack or h[stack[-1]] > height:
                return stack
            return pop_until(stack[:-1], height)

        # Reduce returns (current_stack, list_of_L_values)
        def step(state, idx):
            stack, l_values = state
            new_stack = pop_until(stack, h[idx])
            l_val = new_stack[-1] + 1 if new_stack else 0
            return (new_stack + [idx], l_values + [l_val])

        # Since we can't use loops, we use reduce to iterate through indices
        # However, l_values + [l_val] is O(N), making the whole thing O(N^2).
        # We must use a mutable list and .append() inside the reduce.
        
        l_values = []
        def step_mutable(stack, idx):
            new_stack = pop_until(stack, h[idx])
            l_values.append(new_stack[-1] + 1 if new_stack else 0)
            return new_stack + [idx]
        
        reduce(step_mutable, range(n), [])
        
        # Now we have L_j for each j. 
        # Building j contributes to i in [L_j, j-1].
        # We need to calculate c_i = count of j such that L_j <= i < j.
        # This is equivalent to: for each j, add 1 to range [L_j, j-1].
        # Use a difference array.
        diff = [0] * (n + 1)
        
        # We need to iterate through L_j and j to fill diff.
        # Again, no loops. Use map or reduce.
        def update_diff(s, j):
            l_j = l_values[j]
            # Range is [l_j, j-1]. 
            # Note: indices are 0-based. L_j is already 0-based index.
            # The condition is i < j and no building between i and j is taller than H_j.
            # This means i can be L_j, L_j+1, ..., j-1.
            # But the problem asks for i = 1...N.
            # For a fixed j, the valid i's are those where max(H_{i+1}...H_{j-1}) < H_j.
            # This is true for i from L_j to j-1.
            # (If L_j is the index of the first building to the left taller than H_j)
            # Example: H = [2, 1, 4], j=2 (H_j=4). L_2 = 0. i can be 0, 1.
            # Example: H = [2, 1, 4], j=1 (H_j=1). L_1 = 0. i can be 0.
            
            # Using a mutable list for diff
            if l_j < j:
                diff[l_j] += 1
                diff[j] -= 1
            return s

        reduce(update_diff, range(n), None)
        
        # Prefix sum to get final counts.
        def get_counts(s, i):
            s.append(s[-1] + diff[i])
            return s
        
        # Start prefix sum from index 0
        res = [0]
        reduce(get_counts, range(n), res)
        
        # The result is c_i for i=0...n-1.
        # The prefix sum of diff gives the number of j's for each i.
        # Note: the first element of res is the initial 0.
        # The values we want are res[1]...res[n].
        # But wait, the difference array logic:
        # diff[0] is the start. The value at index i is sum(diff[0...i]).
        # Let's refine:
        
        final_diff = [0] * (n + 1)
        def fill_diff(s, j):
            l_j = l_values[j]
            if l_j < j:
                final_diff[l_j] += 1
                final_diff[j] -= 1
            return s
        reduce(fill_diff, range(n), None)
        
        # Calculate prefix sums
        def accumulate(acc, x):
            acc.append(acc[-1] + x)
            return acc
        
        # We need the sum of final_diff[0...i] for i = 0...n-1
        # But the range of i is 0 to n-1.
        # The result for i is sum(final_diff[0...i]).
        # However, the building j=i cannot be counted (i < j).
        # The difference array already handles this because we add to [L_j, j-1].
        
        # To avoid the loop in the final output, use map and join.
        # We need to compute the prefix sum of final_diff.
        # Since we can't use itertools.accumulate (it's a loop internally, 
        # but the prompt says "no for/while loops", and accumulate is a builtin).
        # Let's use a custom reduce for prefix sum.
        
        prefix_sums = []
        def calc_ps(s, x):
            s.append(s[-1] +