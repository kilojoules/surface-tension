The constraint to avoid explicit loops forces a shift toward functional programming paradigms, which can lead to more concise and declarative code by focusing on "what" to compute rather than "how" to iterate. For this problem, I will use `functools.reduce` to maintain a state (a monotonic stack) while traversing the buildings, and list comprehensions for data transformation.

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
    # means we are looking for elements to the right of i that are 
    # "visible" if we look from i.
    # Specifically, j satisfies the condition if H_k < H_j for all i < k < j.
    # This is equivalent to saying that Building j is a new maximum 
    # encountered while scanning from i+1 to N, ignoring buildings 
    # shorter than the current maximum.
    # However, the condition is simpler: j satisfies it if H_j is 
    # greater than all H_k for i < k < j.
    # This means for a fixed i, we are counting j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements that would remain in a 
    # monotonic increasing stack if we processed the array from i+1 to N.
    
    # To solve this efficiently for all i, we process the array from right to left.
    # For a fixed i, the buildings j that satisfy the condition are:
    # 1. j = i + 1
    # 2. The next building to the right of i+1 that is taller than H_{i+1}
    # 3. The next building to the right of that one that is taller, and so on.
    
    # Let'. Let's use a Segment Tree or a similar structure to find the 
    # next greater element, but since we can't use loops, we can use 
    # a recursive-like structure via reduce.
    # Actually, the condition "no building taller than H_j between i and j"
    # is satisfied if H_j is a prefix maximum of the sequence H_{i+1}, ..., H_N.
    
    # For a fixed i, the answer is the number of prefix maximums of [H_{i+1}, ..., H_N].
    # This is a classic problem that can be solved by building a 
    # Cartesian tree or using a Fenwick tree/Segment tree with binary search.
    # Given the constraints and the "no loop" rule, we can use a 
    # Divide and Conquer approach implemented via recursion.

    def count_visible(l, r, heights):
        if l >= r:
            return [0] * (r - l + 1)
        
        # This is getting complex for a no-loop constraint. 
        # Let's use the property: j satisfies the condition if 
        # H_j > max(H_{i+1} ... H_{j-1}).
        # This means for a fixed i, we are counting indices j > i such that
        # H_j is a left-to-right maximum of the suffix starting at i+1.
        pass

    # Alternative approach: 
    # For each i, we want to count j > i such that max(H_{i+1}...H_{j-1}) < H_j.
    # This is equivalent to: j is a "visible" building from i.
    # Let's use a Segment Tree to find the number of prefix maximums.
    # In a Segment Tree node, we can store the maximum value in the range.
    # We can define a function `query(node, current_max)` that returns the 
    # number of prefix maximums in the range of `node` given the `current_max` 
    # from the left.

    def build_tree(l, r):
        if l == r:
            return {'max': h[l], 'count': 1, 'l': l, 'r': r}
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        return {
            'max': max(left['max'], right['max']),
            'left_child': left,
            'right_child': right,
            'l': l,
            'r': r,
            'res': calc_visible(left, right)
        }

    def calc_visible(left, right):
        # Count how many in the right child are prefix maximums 
        # given the maximum of the left child.
        return count_greater(right, left['max'])

    def count_greater(node, limit):
        if node['max'] <= limit:
            return 0
        if node['l'] == node['r']:
            return 1
        # If the left child's max is <= limit, only the right child can contribute
        if node['left_child']['max'] <= limit:
            return count_greater(node['right_child'], limit)
        # If left child's max > limit, the right child's contribution 
        # is already precomputed relative to the left child's max.
        return count_greater(node['left_child'], limit) + node.get('res', 0)

    # Since we cannot use loops, we use a helper to handle the Segment Tree 
    # logic and map/reduce for the final output.
    # However, the Segment Tree needs to be queried for each i.
    # For i, we need prefix maximums of H[i+1...N-1].
    
    # Let's redefine: we need to query the range [i+1, N-1].
    # A simpler way to implement this without loops is to use 
    # a recursive function with memoization or a Segment Tree.
    
    # Given the constraints and the "no loop" rule, the most idiomatic 
    # way to implement the Segment Tree query is recursion.
    
    def get_ans(tree, l, r, current_max):
        if l > r or tree['max'] <= current_max:
            return 0, current_max
        if tree['l'] == tree['r']:
            return 1, tree['max']
        
        # This is for a range query. But we need the range [i+1, N-1].
        # We can just build the tree for [0, N-1] and query the suffix.
        pass

    # To avoid loops and recursion depth issues, and since N=2e5, 
    # we must be careful. But the problem asks for no loops.
    # Let's use the property: the answer for i is the number of 
    # elements in the monotonic stack when processing H[i+1...N-1].
    # This is still O(N^2) if done naively.
    
    # The most efficient way is the Segment Tree approach.
    # Let's implement it using a functional style.
    
    def solve_recursive():
        # Using a list-based segment tree to avoid object overhead
        # tree_max[v] stores max of range
        # tree_cnt[v] stores number of prefix maxes in right child given left child
        
        # Since we can't use loops, we use recursion to build and query.
        # To avoid recursion limit, we use sys.setrecursionlimit.
        sys.setrecursionlimit(300000)
        
        def build(l, r):
            if l == r:
                return h[l], 1
            mid = (l + r) // 2
            l_max, l_cnt = build(l, mid)
            r_max, r_cnt = build(mid + 1, r)
            
            # We need a way to store the 'count_greater' result.
            # We'll use a dictionary or a custom object.
            return {'max': max(l_max, r_max), 'l': l, 'r': r, 
                    'lc': None, 'rc': None, 'res': 0}

        # Actually, the most straightforward way to implement this 
        # without loops is to use a recursive function that 
        # processes the array and returns the counts.
        
        def count_visible_from(idx, current_max, suffix):
            if idx >= n:
                return 0
            # This is still O(N^2).
            pass

    # Let's use the property: the answer for i is the number of 
    # elements j > i such that H_j > max(H_{i+1}...H_{j-1}).
    # This is equivalent to the number of nodes on the right spine 
    # of the Cartesian tree rooted at the maximum of H[i+1...N-1].
    
    # Final attempt: Use a Segment Tree implemented with recursion.
    # We build the tree once, then for each i, we query the range [i+1, N-1].
    
    def build_st(l, r):
        if l == r:
            return (h[l], 1, None, None, 0) # max, cnt, left, right, precomputed_right_cnt
        mid = (l + r) // 2
        left = build_st(l, mid)
        right = build_st(mid + 1, r)
        
        def get_cnt(node, limit):
            if node is None or node[0] <= limit: return 0
            if node[1] == 1 and node[2] is None: return 1
            # node: (max, cnt, left, right, precomputed_right_cnt)
            # This structure is getting messy. Let's use a class.
            return 0

    # Given the strict "no loop" constraint, the most reliable 
    # way to implement this is using a recursive Segment Tree 
    # and map() for the final output.
    
    class Node:
        def __init__(self, l, r, h_vals):
            self.l = l
            self.r = r
            if l == r:
                self.max = h_vals[l]
                self.cnt = 1
                self.left = self.right = None
            else:
                mid = (l + r) // 2
                self.left = Node(l, mid, h_vals)
                self.right = Node(mid + 1, r, h_vals)
                self.max = max(self.left.max, self.right.max)
                self.cnt = self.calc_right(self.right, self.left.max)

        def calc_right(self, node, limit):
            if node is None or node.max <= limit:
                return 0
            if node.left is None:
                return 1
            if node.left.max <= limit:
                return self.calc_right(node.right, limit)
            return self.calc_right(node.left, limit) + (node.cnt - node.left.cnt if hasattr(node, 'cnt') else 0)
            # This logic is slightly flawed. Let's refine.

    # Correct logic for Segment Tree prefix maximum count:
    # node.sum is the number of prefix maximums in the range [l, r].
    # node.sum = left.sum + count_greater(right, left.max)
    
    class STNode:
        def __init__(self, l, r, h_vals):
            self.l, self.r = l, r
            if l == r:
                self.max = h_vals[l]
                self.sum = 1
                self.left = self.right = None
            else:
                mid = (l + r) // 2
                self.left = STNode(l, mid, h_vals)
                self.right = STNode(mid + 1, r, h_vals)
                self.max = max(self.left.max, self.right.max)
                self.sum = self.left.sum + self.query_greater(self.right, self.left.max)

        def query_greater(self, node, limit):
            if node is None or node.max <= limit:
                return 0
            if node.left is None:
                return 1
            if node.left.max <= limit:
                return self.query_greater(node.right, limit)
            return self.query_greater(node.left, limit) + (node.sum - node.left.sum)

        def query_range(self, ql, qr, limit):
            # Returns (count, new_limit)
            if ql <= self.l and self.r <= qr:
                return self.query_greater_range(self