The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing declarative data transformations over imperative state updates. While not always more "idiomatic" for every task, it encourages the use of higher-order functions and prevents side-effect-driven bugs. To solve this problem, I will use a recursive approach (simulating a stack-based traversal) combined with `map` and `list` comprehensions to calculate the number of visible buildings for each index.

```python
import sys

# Increase recursion depth for deep building lines
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "right-side" visible building from i.
    # This is equivalent to saying Building j is part of the 
    # monotonic increasing sequence of heights starting from i+1.
    
    # To avoid loops, we use a recursive function to process the buildings.
    # We need to find for each i, how many j > i are "visible".
    # A building j is visible from i if H[k] < H[j] for all i < k < j.
    
    # This is equivalent to: j is visible from i if H[j] is greater than 
    # all heights in the range (i, j).
    
    # Let's redefine: for a fixed i, we are looking for the number of 
    # elements in the sequence H[i+1...N-1] that are larger than all 
    # preceding elements in that subsequence.
    
    # However, the constraint is "no building taller than Building j".
    # This means H[k] <= H[j] for all i < k < j.
    # Since all H are distinct, it's H[k] < H[j].
    
    # This is exactly the number of elements that would remain if we 
    # filtered the suffix H[i+1:] to keep only those that are 
    # larger than all elements to their left within that suffix.
    
    # Wait, the condition is: for a fixed i, count j > i such that 
    # max(H[i+1...j-1]) < H[j].
    # This is simply the number of "prefix maximums" of the suffix H[i+1:].
    
    # To solve this for all i without loops, we can use the property:
    # The buildings j that satisfy the condition for i are the buildings
    # that form the upper hull of the heights if we consider them as points.
    # More simply, we can use a Segment Tree or a similar structure, 
    # but recursion/map is required.
    
    # Let's use a Divide and Conquer approach.
    # For a range [L, R], we count pairs (i, j) such that L <= i < j <= R.
    # This is complex to implement without loops.
    
    # Alternative: For each j, it is visible from i if H[j] > max(H[i+1...j-1]).
    # This means i must be such that max(H[i+1...j-1]) < H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # Then for all i from L[j] to j-1, building j is visible from i.
    # (If no building to the left is taller, L[j] = 0).
    # The number of such i is j - L[j].
    
    # We can find L[j] for all j using a monotonic stack.
    # To implement a monotonic stack without loops, we use recursion.
    
    def get_left_taller(idx, stack):
        if idx == N:
            return []
        
        # Remove elements from stack that are smaller than current height
        # Using a helper to filter the stack
        def pop_smaller(s):
            if not s or H[s[-1]] < H[idx]:
                return pop_smaller(s[:-1])
            return s
        
        current_stack = pop_smaller(stack)
        left_taller = current_stack[-1] if current_stack else 0
        
        # We store the result and move to next
        # Note: we use 1-based indexing for the result as per problem L[j]
        # But since we need the count of i, and i is 1-indexed:
        # i can be L[j], L[j]+1, ..., j-1.
        # The number of such i is j - L[j].
        # However, the condition is "between i and j".
        # If i = j-1, there are no buildings between, so it's always true.
        # If i = L[j], the buildings between are H[L[j]+1 ... j-1], 
        # all of which are < H[j] by definition of L[j].
        # If i = L[j]-1, the building at L[j] is between i and j, 
        # and H[L[j]] > H[j], so the condition fails.
        # Thus i ranges from L[j] to j-1.
        # The number of such i is (j-1) - L[j] + 1 = j - L[j].
        # (Using 0-indexing for H, j is the index, L[j] is the index of taller building).
        
        # To avoid creating too many lists, we'll use a different approach for the stack.
        return [left_taller] + get_left_taller(idx + 1, current_stack + [idx])

    # The recursive stack approach above will hit recursion limits and be slow.
    # Let's use a more efficient way to calculate L[j] using a list comprehension
    # and a custom function, but the constraint allows recursion.
    # Actually, the most efficient way to find L[j] is a loop. 
    # Since loops are forbidden, we use a recursive function with a shared list.
    
    L = [0] * N
    def compute_L(idx, stack):
        if idx >= N:
            return
        
        # Simulate while stack and H[stack[-1]] < H[idx]: stack.pop()
        def clean_stack(s):
            if len(s) > 0 and H[s[-1]] < H[idx]:
                return clean_stack(s[:-1])
            return s
        
        s = clean_stack(stack)
        L[idx] = s[-1] + 1 if s else 0
        compute_L(idx + 1, s + [idx])

    # To avoid the O(N^2) slice/concat in the recursion, we use a mutable list
    # and a recursive function that mimics the loop.
    
    stack = []
    def fast_L(idx):
        if idx >= N:
            return
        while stack and H[stack[-1]] < H[idx]:
            stack.pop()
        # Wait, 'while' is a loop. I must use recursion.
        # Let's redefine the 'pop' as a recursive function.
        pass

    # Correcting: The constraint says "no for or while loops".
    # I will use a recursive function to process the array and a 
    # recursive function to handle the stack popping.
    
    res_L = [0] * N
    stk = []
    
    def pop_rec():
        if stk and H[stk[-1]] < H[current_h]:
            stk.pop()
            return pop_rec()
        return None

    # Since I cannot use a loop to iterate through N, I use map() or recursion.
    # But I need to maintain the stack state.
    
    def process(idx):
        if idx == N:
            return
        
        # We need to access H[idx]
        # Using a helper to handle the popping logic
        def handle_pop(i):
            if stk and H[stk[-1]] < H[i]:
                stk.pop()
                return handle_pop(i)
            return None
        
        handle_pop(idx)
        res_L[idx] = stk[-1] + 1 if stk else 0
        stk.append(idx)
        process(idx + 1)

    # To avoid recursion depth issues for N=2e5, we can't use standard recursion.
    # But the prompt forbids loops. This is a contradiction for N=2e5 
    # unless we use specific tricks.
    # However, we can use `functools.reduce` to simulate a loop.
    
    from functools import reduce
    
    def step(s, idx):
        # s is the stack
        # We need to pop elements smaller than H[idx]
        # Since we can't use while, we use a recursive helper
        def pop_small(current_s):
            if current_s and H[current_s[-1]] < H[idx]:
                return pop_small(current_s[:-1])
            return current_s
        
        new_s = pop_small(s)
        # Calculate count for this j: j - L[j]
        # But we need to store this for each i.
        # Let's use a different approach: 
        # For each j, it contributes 1 to c_i for i in [L[j], j-1].
        # This is a range update. We can use a difference array.
        # diff[L[j]] += 1, diff[j] -= 1.
        return new_s + [idx]

    # To implement the range updates without loops:
    # 1. Use reduce to find all L[j]
    # 2. Use a difference array to mark counts
    # 3. Use reduce/itertools.accumulate to get final counts
    
    # Since we can't use a loop to find L[j], we use reduce.
    # To avoid O(N) slicing in pop_small, we can't use lists as stacks in reduce 
    # if we slice them. But we can use a list and mutate it.
    
    # Wait, the constraint says "no for or while loops". 
    # It does NOT forbid mutating a list inside a function called by reduce.
    
    diff = [0] * (N + 1)
    stk_mutable = []
    
    def reducer(s, idx):
        def pop_recursive():
            if stk_mutable and H[stk_mutable[-1]] < H[idx]:
                stk_mutable.pop()
                return pop_recursive()
        
        pop_recursive()
        left_taller = stk_mutable[-1] if stk_mutable else -1
        # i is 0-indexed. i ranges from left_taller + 1 to idx - 1.
        # The number of such i is (idx - 1) - (left_taller + 1) + 1 = idx - left_taller - 1.
        # But we need to add 1 to c_i for i in [left_taller + 1, idx - 1].
        # Using 0-indexing for i:
        # The buildings between i and j are H[i+1 ... j-1].
        # Condition: max(H[i+1 ... j-1]) < H[j].
        # This is true if i+1 > L[j] (where L[j] is index of first taller building to the left).
        # So i > L[j] - 1  => i >= L[j].
        # Also i < j.
        # So i is in range [L[j], j-1].
        # L[j] is the index of the first building to the left of j that is taller than H[j].
        # If no such building, L[j] = -1.
        # Then i can be -1 (not possible), 0, ..., j-1.
        # But i must be >= 0.
        # So i is in [max(0, L[j] + 1), j-1].
        # Wait, if L[j] is the index of the taller building, then any i < L[j] 
        # will have building L[j] between i and j, and H[L[j]] > H[j].
        # So i must be >= L[j]. 
        # But the building at L[j] is NOT between i and j if i = L[j].
        # So i can be L[j], L[j]+1, ..., j-1.
        # However, the problem says i < j.
        # And the buildings between i and j are i+1, ..., j-1.
        # If i = L[j], the