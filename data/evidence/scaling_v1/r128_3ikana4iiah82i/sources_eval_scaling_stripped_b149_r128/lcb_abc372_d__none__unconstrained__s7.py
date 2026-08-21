```python
import sys
from bisect import bisect_right

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We need to find for each i, the number of j > i such that 
    # max(H_{i+1}, ..., H_{j-1}) <= H_j.
    # This is equivalent to counting j > i such that H_j is a 
    # "right-to-left" maximum of the suffix starting at i+1.
    # However, the condition is simpler: j satisfies the condition if 
    # H_j is greater than all heights between i and j.
    # This means if we look at the sequence from j down to i+1, 
    # H_j must be the maximum.
    
    # Let's redefine: for a fixed i, we are looking for j > i such that
    # H_j >= max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the definition of the "upper envelope" or 
    # "increasing subsequence" if we were to pick elements greedily.
    # Actually, the condition "no building taller than Building j between i and j"
    # means H_j >= H_k for all i < k < j.
    
    # This is a classic problem that can be solved by observing that 
    # for a fixed i, the valid j's are the indices of the 
    # prefix maximums of the sequence H[i+1:].
    # But we need this for all i.
    
    # Let's use the property: j is valid for i if H_j is the maximum 
    # of the range [i+1, j].
    # Let L[j] be the largest index k < j such that H_k > H_j.
    # Then for a fixed j, the indices i that satisfy the condition are 
    # those where L[j] <= i < j.
    # The number of such i is j - L[j].
    # Wait, the condition is: for all k such that i < k < j, H_k <= H_j.
    # This means the maximum height in the range (i, j) is <= H_j.
    # Let L[j] be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L[j] = 0.
    # The condition is satisfied if the range (i, j) contains no building taller than H_j.
    # This means the first building taller than H_j to the left must be at index <= i.
    # So L[j] <= i < j.
    # For a fixed i, we want to count j > i such that L[j] <= i.
    
    # Let's find L[j] for all j using a monotonic stack.
    # H is 0-indexed, so buildings are 0 to N-1.
    # L[j] will be the index of the first building to the left of j with height > H[j].
    # If none, L[j] = -1.
    
    stack = []
    l = [0] * n
    for j in range(n):
        while stack and h[stack[-1]] < h[j]:
            stack.pop()
        if not stack:
            l[j] = -1
        else:
            l[j] = stack[-1]
        stack.append(j)
    
    # Now we need to count for each i: how many j > i satisfy L[j] <= i.
    # This is equivalent to: for each j, it contributes to i in range [L[j], j-1].
    # We can use a difference array (or Fenwick tree) to add 1 to range [L[j], j-1].
    # Since we need the result for all i, we can use a technique to 
    # process the queries.
    # However, the range of i is simply L[j] to j-1.
    # We can use a Fenwick tree to perform range updates and point queries.
    # Or, since we can't use loops, we can use a clever trick with 
    # a sorted list of events and a prefix sum.
    
    # Let's use the property: count j > i such that L[j] <= i.
    # This is (total j > i) - (count j > i such that L[j] > i).
    # Total j > i is (n-1) - i.
    # L[j] > i means the first building taller than H[j] to the left is to the right of i.
    
    # Let's use the range update approach with a Fenwick tree.
    # Since we can't use loops, we can use a recursive-like structure or 
    # map/reduce, but the constraints allow list comprehensions.
    # Actually, we can solve this by sorting the L[j] values and using bisect.
    
    # For a fixed i, we want count of j in {i+1, ..., n-1} such that L[j] <= i.
    # This is equivalent to: count j in {0, ..., n-1} such that L[j] <= i AND j > i.
    # Note that L[j] is always < j. So L[j] <= i and j > i is the condition.
    
    # Let's collect all (L[j], j) pairs.
    # We want to count pairs where L[j] <= i < j.
    # This is a 2D range counting problem: L[j] in [-1, i] and j in [i+1, n-1].
    # But since L[j] < j is always true, we only need to count j > i such that L[j] <= i.
    
    # Let's use the fact that we can use bisect on sorted lists.
    # We want to count j such that i < j and L[j] <= i.
    # This is (count j such that L[j] <= i) - (count j such that j <= i and L[j] <= i).
    # Since L[j] < j, the condition (j <= i and L[j] <= i) is simply (j <= i).
    # So for a fixed i, the answer is:
    # (count j in 0..n-1 such that L[j] <= i) - (i + 1).
    
    # Let's verify:
    # For i = 0: (count j such that L[j] <= 0) - 1.
    # If L[j] = -1 or 0, it's counted.
    # For Sample 1: H = [2, 1, 4, 3, 5]
    # j=0: H=2, L[0]=-1
    # j=1: H=1, L[1]=0
    # j=2: H=4, L[2]=-1
    # j=3: H=3, L[3]=2
    # j=4: H=5, L[4]=-1
    # L = [-1, 0, -1, 2, -1]
    # i=0: L[j] <= 0 are j=0, 1, 2, 4. Count = 4. Ans = 4 - 1 = 3. Correct.
    # i=1: L[j] <= 1 are j=0, 1, 2, 4. Count = 4. Ans = 4 - 2 = 2. Correct.
    # i=2: L[j] <= 2 are j=0, 1, 2, 3, 4. Count = 5. Ans = 5 - 3 = 2. Correct.
    # i=3: L[j] <= 3 are j=0, 1, 2, 3, 4. Count = 5. Ans = 5 - 4 = 1. Correct.
    # i=4: L[j] <= 4 are j=0, 1, 2, 3, 4. Count = 5. Ans = 5 - 5 = 0. Correct.
    
    # To implement this without loops:
    # 1. Compute L using a stack (we can use a custom reduce to simulate the stack).
    # 2. Sort L.
    # 3. For each i, use bisect_right to find count of L[j] <= i.
    
    def get_l(heights):
        # Use reduce to build the L array. 
        # State: (stack, results_list)
        from functools import reduce
        def step(state, h_j):
            stack, res = state
            # We need to simulate the while loop to pop the stack.
            # Since we can't use while, we can use a recursive function 
            # or a trick with a list.
            # However, the constraint says no loops. 
            # Let's use a helper to pop the stack.
            def pop_stack(s):
                if s and s[-1][0] < h_j:
                    return pop_stack(s[:-1])
                return s
            
            new_stack = pop_stack(stack)
            l_j = new_stack[-1][0] if new_stack else -1
            # We store (index, height) in stack
            # But wait, the pop_stack needs the height.
            return (new_stack, res + [l_j])
        
        # The above pop_stack is recursive. Let's refine it.
        # Actually, I can use a list comprehension to find the index 
        # of the first element from the right that is > h_j.
        # But that's O(N^2). The stack is necessary.
        pass

    # Let's use a different approach for L.
    # We can use a divide and conquer approach to find L[j].
    # Or use the fact that we can use 'reduce' and 'recursion'.
    # Python's recursion limit is an issue, but we can increase it.
    
    sys.setrecursionlimit(300000)
    
    def compute_l(heights):
        n = len(heights)
        l = [0] * n
        stack = []
        
        def process(j):
            if j == n:
                return
            # Simulate while stack and h[stack[-1]] < h[j]: stack.pop()
            # Since we can't use while, we use a helper.
            def clean_stack(s):
                if s and heights[s[-1]] < heights[j]:
                    return clean_stack(s[:-1])
                return s
            
            curr_stack = clean_stack(stack)
            l[j] = curr_stack[-1] if curr_stack else -1
            stack.extend([j]) # This is a mutation, but allowed in helper
            # Wait, the stack needs to be updated for the next j.
            # Let's use a nonlocal variable for the stack.
            
        # To avoid the loop in process(), we can use map().
        # But the stack depends on the previous state.
        # Let's use reduce.
        return l

    # Correcting the reduce approach:
    from functools import reduce
    
    def solve_with_reduce(n, h):
        def step(state, j):
            stack, l_vals = state
            # Use a helper to pop the stack without a while loop
            def pop_helper(s):
                if s and h[s[-1]] < h[j]:
                    return pop_helper(s[:-1])
                return s
            
            new_stack = pop_helper(stack)
            l_j = new_stack[-1] if new_stack else -1
            return (new_stack + [j], l_vals + [l_j])
        
        # Using a list for l_vals in reduce is O(N^2) due to list concatenation.
        # Let's use a deque or just map the logic.
        # Actually, we can just use a list and mutate it.
        pass

    # Final attempt at a clean, loop-free implementation:
    # 1. Use a helper function with recursion to find L.
    # 2. Use bisect to count.
    
    sys.setrecursionlimit(300000)
    
    # We use a list to store L and a list as a stack.
    # We use a recursive function to iterate through j.
    l = [0] * n
    stack = []
    
    def find_l(j):
        if j == n:
            return
        
        def pop_stack(s):
            if s and