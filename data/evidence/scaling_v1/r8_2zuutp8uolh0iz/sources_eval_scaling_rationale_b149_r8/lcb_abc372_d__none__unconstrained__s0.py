The constraint to avoid explicit loops forces a shift toward functional programming paradigms, which can lead to more concise and declarative code by focusing on "what" to compute rather than "how" to iterate. To solve this problem, I will use a monotonic stack approach implemented via `functools.reduce` to maintain a stack of indices of buildings that could potentially be the "tallest" seen so far, and list comprehensions for the final output.

```python
import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j" 
    # means we are looking for elements to the right of i that are 
    # "visible" if we look from i.
    # Specifically, for a fixed i, j satisfies the condition if 
    # H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that j is an index such that 
    # H[j] is a new maximum encountered while scanning from i+1 to N.
    # However, the problem asks for this for ALL i.
    # Let's rephrase: for a fixed j, it satisfies the condition for i if
    # H[j] > max(H[i+1を...j-1]).
    # This means for a fixed j, it contributes to all i < j such that 
    # there is no k (i < k < j) with H[k] > H[j].
    # This is equivalent to saying i must be greater than the index of the 
    # first building to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # Then j satisfies the condition for all i in the range [L[j], j-1].
    # The number of such i is j - L[j].
    # Wait, the condition is: "no building taller than Building j between i and j".
    # If L[j] is the index of the first building to the left of j taller than H[j],
    # then for any i < L[j], Building L[j] is between i and j and is taller than Building j.
    # For any i >= L[j] (and i < j), no building between i and j is taller than Building j.
    # So for a fixed j, the valid i's are L[j], L[j]+1, ..., j-1.
    # Note: if no such L[j] exists, we can treat L[j] as 0 (1-indexed).
    # The number of such i is j - L[j].
    
    # We need to calculate c_i = count of j > i satisfying the condition.
    # This is sum_{j=i+1}^N [i >= L[j]]
    # Let's use a monotonic stack to find L[j] for all j.
    # stack stores indices.
    
    def find_L(acc, x):
        stack, L = acc
        # Pop elements smaller than current height
        while stack and H[stack[-1]] < H[x]:
            stack.pop()
        # L[x] is the index of the first element taller than H[x]
        # If stack is empty, L[x] = -1 (0-indexed)
        current_L = stack[-1] if stack else -1
        return (stack + [x], L + [current_L])

    # Using reduce to simulate the monotonic stack
    # Initial state: (stack, L_list)
    final_state = reduce(lambda acc, x: (
        # We can't use a while loop, so we use a helper function or 
        # a recursive-like structure. Since we can't use loops, 
        # we must find a way to handle the stack.
        # Actually, the constraint says "no for or while loops".
        # I will use a recursive function to handle the stack popping.
        pass 
    ), range(N), ([], []))
    # Since the above is complex without loops, let's redefine.

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # To avoid loops and recursion (which might hit limits), 
    # we can use the property that we need to count j > i such that 
    # max(H[i+1...j-1]) < H[j].
    # This is equivalent to: for each j, it is counted for i in [L[j], j-1]
    # where L[j] is the index of the first building to the left taller than H[j].
    
    # Since I cannot use loops, I will use a recursive function for the stack
    # and increase the recursion depth.
    sys.setrecursionlimit(300000)
    
    def get_L(stack, idx):
        if idx == N:
            return []
        # Remove elements from stack that are smaller than H[idx]
        # Using a helper to simulate the while loop via recursion
        def pop_smaller(s):
            if s and H[s[-1]] < H[idx]:
                return pop_smaller(s[:-1])
            return s
        
        new_stack = pop_smaller(stack)
        left_val = new_stack[-1] if new_stack else -1
        return [left_val] + get_L(new_stack + [idx], idx + 1)

    # However, the most reliable way to avoid loops in Python for this 
    # specific problem (monotonic stack) while staying within limits 
    # is to use a custom reduce or a mapping.
    # But the "no loop" constraint is strict. 
    # Let's use a different approach: 
    # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements that would be added to a 
    # monotonic increasing stack if we processed the array from i+1 to N.
    
    # Let's use the L[j] logic. L[j] is the index of the nearest element to the left > H[j].
    # We can find L[j] using a recursive function.
    def compute_L(idx, stack):
        if idx == N: return []
        # Simulate while stack and H[stack[-1]] < H[idx]: stack.pop()
        def clean_stack(s):
            return clean_stack(s[:-1]) if s and H[s[-1]] < H[idx] else s
        
        s_now = clean_stack(stack)
        res = s_now[-1] if s_now else -1
        return [res] + compute_L(s_now + [idx], idx + 1)

    # To avoid recursion depth and loops, we can use a trick with 
    # a list and a reduce that manages the stack in a way that 
    # doesn't require explicit loops.
    # But wait, the most idiomatic "no loop" way to find the nearest 
    # greater element is using a Segment Tree or Fenwick Tree, 
    # but those usually require loops for updates.
    
    # Let's use the property: c_i is the number of j > i such that 
    # H[j] is a prefix maximum of the sequence H[i+1...N].
    # This is a known problem that can be solved by a Segment Tree.
    # In a Segment Tree, we can store the maximum of a range and 
    # a function count(node, current_max) that returns how many 
    # prefix maximums are in the range given the maximum to the left.
    
    # Since I cannot use loops, I will use a recursive Segment Tree.
    def build(l, r):
        if l == r: return (H[l], 1)
        mid = (l + r) // 2
        left = build(l, mid)
        right = build(mid + 1, r)
        # The number of prefix maximums in the right child depends on the max of the left child.
        return (max(left[0], right[0]), 0) # Placeholder

    # This is getting complex. Let's use the L[j] logic with a 
    # functional approach to build the L array.
    # We can use a recursive function to find L[j] and then 
    # use a Fenwick tree (implemented recursively) to count.
    # Actually, the simplest way to implement the monotonic stack 
    # without loops is using a recursive function.
    
    sys.setrecursionlimit(300000)
    def get_L_recursive(idx, stack):
        if idx == N: return []
        # Use a helper to pop
        def pop_func(s):
            if not s or H[s[-1]] > H[idx]: return s
            return pop_func(s[:-1])
        
        s_next = pop_func(stack)
        val = s_next[-1] if s_next else -1
        return [val] + get_L_recursive(idx + 1, s_next + [idx])

    # The number of i's for which j is a valid index is j - L[j].
    # We need c_i = sum_{j=i+1}^{N-1} [L[j] <= i]
    # This is a range update point query problem.
    # For each j, increment range [L[j], j-1].
    # Then query values at each i.
    # This can be done by:
    # diff[L[j]] += 1, diff[j] -= 1
    # Then prefix sum of diff.
    
    # To implement this without loops:
    # 1. Get L array via recursion.
    # 2. Create diff array.
    # 3. Use reduce to compute prefix sums.
    
    # Since L[j] can be -1, we use 0-indexing carefully.
    # L[j] is the index of the first building to the left > H[j].
    # Valid i are L[j] <= i < j. (Using 0-indexing)
    # Note: if L[j] is -1, then 0 <= i < j.
    
    # To avoid the O(N^2) behavior of s[:-1], we use a list and 
    # a pointer or just accept that we must use a different approach.
    # Actually, the most efficient way to find L[j] without loops 
    # is to use a divide and conquer approach.
    
    def solve_dc(l, r):
        if l == r: return [0] * 1 # Not useful for c_i
        # This is also complex.
        pass

# Given the constraints and the "no loop" rule, the most 
# viable path is using recursion for the monotonic stack 
# and reduce for the prefix sums.
# To avoid O(N^2) with slices, I'll use a list and a 
# recursive function that passes the stack.

def final_solve():
    import sys
    sys.setrecursionlimit(300000)
    input_data = sys.stdin.read().split()
    if not input_data: return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need L[j]: index of nearest element to the left > H[j]
    # Using a list as a stack and a recursive function to simulate the while loop
    stack = []
    def get_L(idx):
        if idx == N: return []
        
        def pop_elements():
            if stack and H[stack[-1]] < H[idx]:
                stack.pop()
                return pop_elements()
            return None
        
        pop_elements()
        res = stack[-1] if stack else -1
        stack.append(idx)
        return [res] + get_L(idx + 1)

    L = get_L(0)
    
    # c_i = count j > i such that L[j] <= i
    # For each j, it contributes to i in range [max(0, L[j]), j-1]
    # Use a difference array to count
    diff = [0] * (N + 1)
    
    # We need to iterate through L to fill diff. 
    # Since we can't use for loops, we use a recursive function.
    def fill_diff(idx):
        if idx == N: return
        # j = idx, L[j] = L[idx