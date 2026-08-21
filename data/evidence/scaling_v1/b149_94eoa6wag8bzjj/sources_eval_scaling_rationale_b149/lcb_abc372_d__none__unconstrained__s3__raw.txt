The constraint to avoid explicit loops forces a functional programming style, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without `for` or `while` loops, I will use `functools.reduce` to maintain a state (a monotonic stack) as I iterate through the buildings from right to left.

For a fixed $i$, we are looking for $j > i$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is equivalent to counting how many buildings to the right are "visible" if we look from building $i$. Specifically, if we process the array from right to left, the buildings that satisfy this condition for index $i$ are exactly the elements of a strictly increasing stack of heights encountered so far.

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
    
    # We process the buildings from right to left.
    # We maintain a stack of buildings that could be 'visible'.
    # For a building i, the buildings j > i that satisfy the condition
    # are those that would form a strictly increasing sequence of heights
    # when looking from i to the right.
    # However, the condition "no building taller than H_j between i and j"
    # means we are looking for the number of elements in a monotonic stack
    # constructed from the right.
    
    # Specifically, for index i, we want to count j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements in a stack where we push H_j
    # if it's taller than all elements added after it.
    # Wait, the condition is: H_j is taller than everything between i and j.
    # This means if we are at i, and we look at the sequence to the right,
    # the first building (i+1) is always counted. The next is the first building
    # taller than H_{i+1}, the next is the first building taller than that, and so on.
    
    # To implement this without loops, we use reduce to traverse the list backwards.
    # State: (stack, results)
    # For each height x, the number of visible buildings to the right is the size
    # of the monotonic stack formed by elements to the right of x.
    # But the stack depends on the starting point i. 
    # Actually, the buildings j that satisfy the condition for i are the 
    # "upper envelope" of the heights to the right.
    # If we maintain a stack of heights from right to left that is strictly 
    # increasing (from the perspective of the right end), that's not quite it.
    
    # Correct logic: For a fixed i, the valid j's are:
    # j1 = i + 1
    # j2 = first index > j1 such that H_{j2} > H_{j1}
    # j3 = first index > j2 such that H_{j3} > H_{j2}...
    # This is equivalent to the number of elements in a monotonic stack 
    # if we process from right to left and keep only elements that are 
    # larger than everything to their right. 
    # No, that's not right. Let's re-evaluate.
    # For i, j is valid if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means H_j must be a prefix maximum of the sequence [H_{i+1}, ..., H_N].
    # The number of such j is the number of prefix maximums.
    
    # To solve this for all i efficiently:
    # We can use a Segment Tree or a similar structure to find the next 
    # greater element, but that's complex without loops.
    # Alternatively, observe that the number of prefix maximums for i 
    # is 1 + (number of prefix maximums for i+1 starting from the first 
    # element taller than H_{i+1}).
    
    # Let's use the property: the answer for i is the number of elements 
    # in the monotonic stack (strictly increasing) created by processing 
    # the array from index i+1 to N.
    # Since we need this for all i, we can use a Segment Tree to query 
    # the number of prefix maximums. 
    # A known trick for "range prefix maximum count" queries:
    # In a segment tree node, store the max value of the range.
    # The function `count(node, current_max)` returns the number of 
    # prefix maximums in `node`'s range given the maximum seen so far.
    
    # However, implementing a Segment Tree without loops/recursion 
    # (recursion is allowed, but loops aren't) is tricky.
    # Let's use the fact that we can use a Fenwick tree or Segment Tree 
    # with list comprehensions and map.
    
    # Wait, the constraint to avoid loops is strict. 
    # Let's use a recursive function for the Segment Tree.
    
    def build_tree(l, r):
        if l == r:
            return {'max': h[l], 'size': 1, 'l': None, 'r': None}
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        return {'max': max(left['max'], right['max']), 'l': left, 'r': right}

    def query_prefix_max(node, current_max):
        if node is None: return 0
        if node['max'] <= current_max: return 0
        # If it's a leaf
        if node['l'] is None and node['r'] is None:
            return 1 if node['max'] > current_max else 0
        
        # If the left child's max is <= current_max, 
        # everything in the left is ignored, and we check the right.
        if node['l']['max'] <= current_max:
            return query_prefix_max(node['r'], current_max)
        
        # If left child's max > current_max, the right child's 
        # contribution is pre-calculable.
        # The number of prefix maximums in the right child relative to 
        # the left child's max is:
        # (total prefix maxes in node) - (prefix maxes in left child)
        # But we need to store the pre-calculated count in the node.
        return 0 # Placeholder

    # Let's refine the Segment Tree to store the pre-calculated 
    # count of prefix maximums in the right child relative to the left.
    def build_refined(l, r):
        if l == r:
            return {'max': h[l], 'cnt': 1, 'l': None, 'r': None}
        mid = (l + r) // 2
        left = build_refined(l, mid)
        right = build_refined(mid + 1, r)
        # cnt is the number of prefix maximums in the right subtree 
        # given the maximum of the left subtree.
        def get_cnt(node, cur_max):
            if node is None: return 0
            if node['max'] <= cur_max: return 0
            if node['l'] is None: return 1
            if node['l']['max'] <= cur_max:
                return get_cnt(node['r'], cur_max)
            return get_cnt(node['l'], cur_max) + (node['cnt'] - node['l']['cnt'] if 'pre' in node else 0)
            # This is getting complex. Let's use a simpler approach.
            
    # Let's use the property: the answer for i is the number of 
    # elements in the monotonic stack of the suffix [i+1, N].
    # We can use a Divide and Conquer approach.
    def solve_recursive(l, r):
        if l == r:
            return [0]
        mid = (l + r) // 2
        # This is also complex.
        
    # Final attempt: Use a simple recursive function with memoization 
    # or a Segment Tree implemented carefully.
    # Actually, the most straightforward way to count prefix maximums 
    # in a range is a Segment Tree where each node stores the max of its range 
    # and a precomputed value: "how many prefix maximums are in the right 
    # child if the left child's maximum is used as the threshold".
    
    def build(l, r):
        if l == r:
            return {'max': h[l], 'res': 1, 'l': None, 'r': None}
        mid = (l + r) // 2
        left = build(l, mid)
        right = build(mid + 1, r)
        # The 'res' for a node is the number of prefix maximums in the 
        # range [l, r].
        # res = left.res + count_prefix_max(right, left.max)
        def count_p(node, limit):
            if node is None: return 0
            if node['max'] <= limit: return 0
            if node['l'] is None: return 1
            if node['l']['max'] <= limit:
                return count_p(node['r'], limit)
            return count_p(node['l'], limit) + (node['res'] - left_node_res_at_build(node))
            # This requires storing the left child's contribution.
        return {}

    # Given the constraints and the "no loop" rule, the most reliable 
    # way to implement this is using a Segment Tree with a helper 
    # function to calculate the prefix maximums.
    
    def get_count(node, limit):
        if node is None: return 0
        if node['max'] <= limit: return 0
        if node['l'] is None: return 1
        if node['l']['max'] <= limit:
            return get_count(node['r'], limit)
        return get_count(node['l'], limit) + node['right_cnt']

    def build_final(l, r):
        if l == r:
            return {'max': h[l], 'right_cnt': 0, 'l': None, 'r': None}
        mid = (l + r) // 2
        left = build_final(l, mid)
        right = build_final(mid + 1, r)
        # right_cnt is get_count(right, left['max'])
        return {
            'max': max(left['max'], right['max']),
            'right_cnt': get_count(right, left['max']),
            'l': left,
            'r': right
        }

    if n == 0: return
    root = build_final(0, n - 1)
    
    def query(l, r, node, current_l, current_r):
        # We need prefix maximums in range [l, r]
        # Since we always query [i+1, n-1], we can just use a 
        # modified get_count.
        pass

    # For each i, we want prefix maximums of h[i+1 : n]
    # We can't easily query arbitrary ranges with get_count.
    # But we only need suffixes!
    # For a suffix, we can just traverse the tree.
    
    def query_suffix(node, l, r, ql, qr, current_max):
        if ql <= l and r <= qr:
            # Return (count, new_max)
            # But get_count only returns count.
            # We need to know the max of the range to pass to the next.
            pass

    # Let's use a simpler approach: the answer for i is 
    # get_count(root_of_range_[i+1, n-1], -1)
    # Since we can't build N trees, we can use the Segment Tree 
    # and a recursive function that collects the count.
    
    def get_suffix_count(node, l, r, ql, qr, current_max):
        if node is None or r < ql or l > qr:
            return 0, current_max
        if ql <= l and r <= qr:
            # This is the tricky part. We need the count of prefix maxes
            # in this node given current_max.
            # We can use the get_count logic.
            # But we need to return the updated current_max.
            # However, the max of the range is already stored.
            # The number of prefix maxes in this node given current_