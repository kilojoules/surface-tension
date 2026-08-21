The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing the transformation of data streams over the mutation of variables. To solve this problem, I will use `itertools.accumulate` to maintain a running state (a stack of indices) as I traverse the buildings, and list comprehensions to process the results.

```python
import sys
from itertools import accumulate

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j" 
    # means we are looking for the number of indices j > i such that 
    # H_j is a prefix maximum of the sequence H_{i+1}, ..., H_N.
    # However, a simpler observation is: for a fixed i, we want to count j > i
    # such that for all k (i < k < j), H_k < H_j.
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many elements to the right are "visible"
    # if we look from index i.
    # Actually, the condition is: j satisfies the condition if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means for a fixed i, the valid j's are the indices of the 
    # running maximums of the suffix starting at i+1.
    
    # Let's redefine: for each i, we want to count j > i such that 
    # H_j is greater than all elements between i and j.
    # This is exactly the number of elements in the "monotonic stack" 
    # if we were to process the array from i+1 to N.
    # But we need this for all i.
    
    # Key Insight: 
    # An index j is "counted" for index i if H_j is a prefix maximum of the range [i+1, N].
    # Let's use a stack-based approach to find for each j, the range of i's it serves.
    # Building j is a prefix maximum for all i such that max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then building j satisfies the condition for all i such that L[j] <= i < j.
    # Note: the condition says "between i and j", so if i = L[j], the building at L[j] 
    # is NOT between i and j. Thus i can be L[j].
    # The number of such i for a fixed j is j - L[j].
    # We want for each i, the count of j > i such that L[j] <= i.
    
    # To implement this without loops:
    # 1. Find L[j] for all j using a stack and accumulate.
    # 2. Use a difference array (or similar) to count how many j's cover each i.
    
    # Finding L[j] using accumulate:
    # State: (stack, result_L)
    def find_L(state, current_h):
        stack, res = state
        # Remove elements smaller than current_h
        # Since we can't loop, we use a trick with a helper function or 
        # accept that the stack logic is hard without loops.
        # Wait, the constraint says no for/while loops. 
        # We can use a recursive-like structure via accumulate if we are clever.
        pass

    # Actually, the most idiomatic way to find the nearest greater element 
    # without loops is using a stack inside a function, but the prompt 
    # forbids loops entirely. We can use a Divide and Conquer approach 
    # implemented via recursion (which is allowed).
    
    def get_nearest_greater_left(heights):
        # Using a recursive approach to simulate the monotonic stack
        # However, recursion depth is an issue. Let's use a different approach.
        # We can use the property that we want to count j > i such that L[j] <= i.
        # This is equivalent to: for each i, count j in (i, N] such that L[j] <= i.
        pass

    # Let's reconsider: j is counted for i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is true if and only if there is no k in (i, j) such that H_k > H_j.
    # This means i >= L[j], where L[j] is the index of the nearest building 
    # to the left of j that is taller than H_j.
    
    # To find L[j] without loops:
    # We can use the fact that H_i are distinct and in range [1, N].
    # We can process heights in increasing order.
    # For the current height H_j, L[j] is the max index in the range [1, j-1] 
    # that has already been processed? No, that's for smaller.
    # For the current height H_j, L[j] is the nearest index to the left 
    # that is TALLER. So we need to process heights in DECREASING order.
    
    # When we process H_j in decreasing order, all buildings currently "active" 
    # are taller than H_j. L[j] is the active index immediately to the left of j.
    # We can use a SortedList or a Fenwick tree, but those require loops to initialize.
    # Actually, we can use a Segment Tree implemented recursively.
    
    # But there is a simpler way. 
    # The number of j > i satisfying the condition is simply:
    # (Number of j > i) - (Number of j > i such that there is some k in (i, j) with H_k > H_j).
    # This is still complex.
    
    # Let's use the property: j is counted for i if H_j is a prefix maximum of H[i+1:].
    # This is a classic problem. The answer for i is the number of 
    # "visible" buildings looking right from i.
    # This can be solved by a Segment Tree. For a range, the number of visible 
    # buildings is:
    # count(left_child) + count_visible(right_child, max_of_left_child)
    
    def build_tree(l, r):
        if l == r:
            return (h[l], 1, 1) # (max_val, count, length)
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        
        def calc_visible(node, threshold):
            if node[0] <= threshold: return 0
            if node[2] == 1: return 1
            # If the left child's max is <= threshold, only the right child can have visible
            # This requires the tree to store more info.
            return 0 # Placeholder
            
    # Since I cannot use loops, I will use recursion for the Segment Tree.
    # Each node will store: max height in range, and a function/value 
    # to calculate visible elements given a threshold.
    
    def solve_recursive():
        # Using a Segment Tree where each node stores the max height 
        # and the number of elements that would be visible if 
        # looked at from the left, given the max of the preceding range.
        
        def get_visible(l, r, threshold, node_idx):
            # This is the standard Segment Tree approach for "Range Maximum Query" 
            # combined with "Prefix Maximum Count".
            pass
            
    # Given the constraints and the "no loop" rule, the most reliable 
    # way to implement this is using a recursive Segment Tree.
    
    # To avoid loops for the final output, I'll use a list comprehension.
    # To avoid loops for the tree, I'll use recursion.
    
    # However, the most direct interpretation of "no building taller than 
    # Building j between i and j" is that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements that would be added to a 
    # monotonic stack when processing the suffix H[i+1:].
    
    # Let's use the property: the answer for i is the number of 
    # indices j > i such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to: 1 + (answer for i+1, but only counting 
    # elements taller than H_{i+1}).
    # This looks like a functional recurrence.
    
    # Let f(i, threshold) be the number of j > i such that 
    # H_j > max(threshold, H_{i+1}, ..., H_{j-1}).
    # The answer for i is f(i, -1).
    # f(i, threshold) = 
    #   if H_{i+1} > threshold: 1 + f(i+1, H_{i+1})
    #   else: f(i+1, threshold)
    
    # This is still O(N^2) if not careful. But we can optimize it 
    # using a Segment Tree.
    # For a range [L, R], we store the max height.
    # We also store 'count', the number of visible buildings in [L, R] 
    # when viewed from the left of L.
    
    # To implement this without loops:
    def build(l, r):
        if l == r:
            return {'max': h[l], 'cnt': 1, 'l': l, 'r': r}
        mid = (l + r) // 2
        left = build(l, mid)
        right = build(mid + 1, r)
        return {
            'max': max(left['max'], right['max']),
            'cnt': left['cnt'] + query_visible(right, left['max']),
            'left': left,
            'right': right,
            'l': l,
            'r': r
        }

    def query_visible(node, threshold):
        if node['max'] <= threshold:
            return 0
        if node['l'] == node['r']:
            return 1
        # If the left child's max is <= threshold, 
        # the left child is invisible, and we check the right child.
        if node['left']['max'] <= threshold:
            return query_visible(node['right'], threshold)
        # If the left child's max is > threshold, 
        # the number of visible elements in the right child is 
        # already precomputed as (node['cnt'] - node['left']['cnt']).
        return query_visible(node['left'], threshold) + (node['cnt'] - node['left']['cnt'])

    # The answer for index i is query_visible(root, h[i]) 
    # but we need to handle the range [i+1, N-1].
    # So we can build the tree for [0, N-1] and for each i, 
    # we query the range [i+1, N-1].
    # But querying a range in this kind of Segment Tree is tricky.
    # Alternatively, we can just realize that the answer for i 
    # is the number of visible elements in the suffix [i+1, N-1].
    # We can compute this for all i by traversing the tree.
    
    # Let's use a simpler approach: 
    # The answer for i is the number of j > i such that H_j > max(H_{i+1}...H_{j-1}).
    # This is exactly the number of elements in the monotonic stack 
    # after processing the suffix.
    # Wait, the most efficient way is:
    # For each j, it is visible from i if i >= L[j].
    # So for a fixed i, we count j > i such that L[j] <= i.
    # This is: (count j > i) - (count j > i such that L[j] > i).
    # Since L[j] < j, the condition L[j] > i means i < L[j] < j.
    
    # Let's use the recursive Segment Tree to find L[j] for all j.
    # L[j] is the nearest index to the left with