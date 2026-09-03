```python
import sys

def solve():
    # Read N from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] <= H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking to the right from Building i, but specifically 
    # based on the height of the target building j itself.
    #
    # Actually, the condition is: j satisfies the condition if 
    # max(H[i+1]...H[j-1]) < H[j].
    # This means Building j must be taller than all buildings between i and j.
    # This is exactly the definition of buildings that would form a 
    # strictly increasing subsequence of "running maximums" starting from i+1.
    #
    # However, the constraint to avoid explicit loops forces a functional 
    # programming style, which encourages the use of higher-order functions 
    # and comprehensions, often leading to more concise code but sometimes 
    # less idiomatic Python for complex algorithms.
    #
    # For this specific problem, we can observe that for a fixed i, 
    # the valid j's are the indices of the prefix maximums of the array H[i+1:].
    # But calculating this for all i would be O(N^2).
    # We need a more efficient approach.
    #
    # Let's rephrase: j is counted for i if H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying that there is no k such that i < k < j and H[k] > H[j].
    # In other words, for a fixed j, it is counted for all i < j such that 
    # for all k in (i, j), H[k] < H[j].
    # This means i must be greater than the index of the first building to the left 
    # of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the valid i's are i = L[j], L[j]+1, ..., j-1.
    # Wait, the condition is "between i and j". 
    # If i = L[j], the buildings between i and j are indices L[j]+1 ... j-1.
    # All these are shorter than H[j] by definition of L[j].
    # So i can be any value from L[j] to j-1.
    # The number of such i's is j - L[j].
    #
    # But we need the count for each i.
    # For a fixed i, we want the number of j > i such that L[j] <= i.
    # This is equivalent to counting j in range [i+1, N] such that L[j] <= i.
    
    # To implement this without loops:
    # 1. Compute L[j] for all j using a stack (but we can't use loops).
    # Since we can't use loops, we can use a recursive function with a 
    # functools.reduce or a similar construct to simulate the stack.
    
    from functools import reduce

    # Simulate stack-based L[j] calculation using reduce
    # stack stores indices of buildings
    # state: (stack, L_list)
    def compute_L(acc, current_idx):
        stack, L_list = acc
        # We need to pop from stack while H[stack[-1]] < H[current_idx]
        # Since we can't use while, we can use a helper function or 
        # a list comprehension trick, but that's tricky.
        # Actually, the constraint says "no for/while loops". 
        # We can use recursion.
        
        def pop_stack(s, val):
            if not s or H[s[-1]] > val:
                return s
            return pop_stack(s[:-1], val)
        
        new_stack = pop_stack(stack, H[current_idx])
        left_boundary = new_stack[-1] + 1 if new_stack else 1
        return (new_stack + [current_idx], L_list + [left_boundary])

    # However, Python's recursion limit is an issue for N=2e5.
    # Let's use a different approach for L[j].
    # We can use a Segment Tree or Fenwick Tree to count j's, 
    # but we still need to compute L[j].
    # Wait, the constraint to avoid loops is very strict. 
    # Let's use a Divide and Conquer approach implemented via recursion 
    # (increasing sys.setrecursionlimit).
    
    sys.setrecursionlimit(300000)
    
    # For a fixed i, we want to count j > i such that max(H[i+1...j-1]) < H[j].
    # This is exactly the number of elements in the "Right-Side Visible" set.
    # This is a classic problem solvable by a Segment Tree.
    # For a range [L, R], the number of visible elements from the left 
    # is calculated by comparing the max of the left child with the right child.
    
    # Since I cannot use loops, I will use recursion and list comprehensions.
    # To avoid the "no loop" constraint while maintaining efficiency, 
    # I'll use a Segment Tree structure.
    
    def build_tree(l, r):
        if l == r:
            return (H[l], 1) # (max_val, count)
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        return (max(left[0], right[0]), count_visible(left[0], right))

    def count_visible(max_val, node):
        # node is (max_val_of_range, count_of_visible_from_left)
        # This is the core of the "Segment Tree Beats" / "Range Maximum Query" 
        # logic for counting visible elements.
        # If the max of the left range is greater than the max of the right range,
        # the right range's contribution depends on the left max.
        pass
    
    # Actually, the simplest way to solve this is:
    # For each i, the answer is the number of j > i such that H[j] is a 
    # prefix maximum of the sequence H[i+1...N].
    # This is equivalent to: 
    # ans[i] = 1 + (ans[next_greater_element[i+1]]) if i+1 < N else 0.
    # Let G[i] be the index of the first building j > i such that H[j] > H[i].
    # Then the buildings visible from i are:
    # Building G[i+1], and then the buildings visible from G[i+1].
    # So c[i] = 1 + c[G[i+1]] (if G[i+1] exists).
    # Special case: Building i+1 is always visible.
    # So c[i] = 1 + (c[G[i+1]] if G[i+1] <= N else 0).
    
    # To find G[i] without loops:
    # We can use the property that G[i] can be found by checking G[i+1], G[G[i+1]], etc.
    # But that's a loop. We can use recursion.
    
    # Let's use the property: G[i] is the first j > i with H[j] > H[i].
    # We can compute G[i] for all i from N down to 1.
    # Since we can't use loops, we use a recursive function with memoization.
    
    memo_G = {}
    def get_G(i):
        if i >= N - 1:
            return N + 1
        if i in memo_G:
            return memo_G[i]
        
        # Start searching from i+1
        curr = i + 1
        # We need to find the first curr such that H[curr] > H[i]
        # We can jump using the G values we already computed.
        def find_next(c):
            if c >= N:
                return N + 1
            if H[c] > H[i]:
                return c + 1 # Using 1-based indexing for the result
            return find_next(get_G(c) - 1)
        
        res = find_next(curr)
        memo_G[i] = res
        return res

    # The above is still recursive. Let's use a more robust approach.
    # The constraint to avoid loops is likely to force the use of 
    # high-level abstractions. However, for N=2e5, recursion is risky.
    # But the prompt says "no for/while loops". 
    # I will use a recursive function to compute G and then C.
    
    # To avoid RecursionError, I'll use a trick with `map` and `reduce` 
    # to simulate the process or just set the limit very high.
    
    # Let's refine the G[i] logic:
    # G[i] = index of first j > i such that H[j] > H[i].
    # If we process i from N-1 down to 0:
    # G[i]: start with j = i + 1. While j < N and H[j] <= H[i]: j = G[j].
    
    # Since I must avoid loops, I will use a recursive function for G 
    # and a recursive function for C.
    
    # To handle the "no loop" constraint and the N=2e5 limit:
    # I will use a stack-based approach implemented via `reduce`.
    
    def get_G_reduce(H):
        # Returns G array where G[i] is the index of the first j > i with H[j] > H[i]
        # We process from right to left.
        # Stack stores indices of elements that could be the next greater element.
        def step(acc, i):
            stack, G = acc
            def pop_smaller(s):
                if not s or H[s[-1]] > H[i]:
                    return s
                return pop_smaller(s[:-1])
            
            s_cleaned = pop_smaller(stack)
            g_val = s_cleaned[-1] if s_cleaned else N
            return (s_cleaned + [i], [g_val] + G)
        
        # Using a custom recursive pop because we can't use while.
        # But s[:-1] creates new lists, making it O(N^2). 
        # This is不 acceptable.
        pass

    # Let's reconsider: the only way to avoid loops and O(N^2) 
    # is to use built-in functions that iterate (like map, reduce, filter, 
    # or list comprehensions) or recursion.
    # But list comprehensions cannot maintain state (like a stack).
    
    # Wait, the most efficient way to solve this is:
    # For each i, the answer is the number of elements in the 
    # "monotonic chain" starting from i+1.
    # Let dp[j] be the number of elements in the monotonic chain starting at j.
    # dp[j] = 1 + dp[G[j]] (if G[j] < N) else 1.
    # The answer for i is dp[i+1] if i+1 < N else 0.
    # But this is only if H[i+1] is the first element.
    # Actually, the buildings visible from i are those j > i such that 
    # H[j] > max(H[i+1...j-1]).
    # This means j=i+1 is always visible, and the next visible is G[i+1], 
    # then G[G[i+1]], and so on.
    # So c[i] = 1 + dp[G[i+1]] where dp[j] is the length of the chain.
    
    # To compute G[i] without loops and without O(N^2) slicing:
    # We can use the "Divide and Conquer" approach to find the Next