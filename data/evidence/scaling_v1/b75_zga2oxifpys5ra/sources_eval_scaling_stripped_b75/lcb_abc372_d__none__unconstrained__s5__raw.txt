```python
import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking right from Building i.
    # A building j is visible from i if it is a left-to-right maximum of the 
    # sequence H[i+1], H[i+2], ..., H[N-1].
    
    # To solve this efficiently for all i, we can use a Segment Tree or a 
    # Divide and Conquer approach. However, the problem can be framed as:
    # For a range [L, R], how many j in [L+1, R] are maximums of [L+1, j]?
    # This is a classic problem solvable by a Segment Tree where each node
    # stores the maximum height in its range and a pre-calculated count of
    # visible buildings from the left of that range given a certain height.

    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.tree_max = [0] * (4 * self.n)
            self.tree_count = [0] * (4 * self.n)
            self._build(data, 1, 0, self.n - 1)

        def _build(self, data, node, start, end):
            if start == end:
                self.tree_max[node] = data[start]
                self.tree_count[node] = 1
                return
            mid = (start + end) // 2
            self._build(data, 2 * node, start, mid)
            self._build(data, 2 * node + 1, mid + 1, end)
            self.tree_max[node] = max(self.tree_max[2 * node], self.tree_max[2 * node + 1])
            self.tree_count[node] = self._calc(2 * node + 1, mid + 1, end, self.tree_max[2 * node])

        def _calc(self, node, start, end, limit):
            # Returns number of elements in range [start, end] that are 
            # left-to-right maximums, given that the previous maximum was 'limit'.
            if self.tree_max[node] <= limit:
                return 0
            if start == end:
                return 1 if self.tree_max[node] > limit else 0
            
            mid = (start + end) // 2
            # If left child's max is <= limit, only the right child can contribute
            if self.tree_max[2 * node] <= limit:
                return self._calc(2 * node + 1, mid + 1, end, limit)
            else:
                # Left child contributes, and right child's contribution is 
                # already pre-calculated relative to left child's max.
                # Total = calc(left, limit) + (total_count_of_node - count_of_left_child)
                # Wait, the pre-calculated count in the node is exactly the 
                # contribution of the right child given the left child's max.
                return self._calc(2 * node, start, mid, limit) + (self.tree_count[node] - self.tree_count[2 * node])
                # Correction: the logic above is slightly flawed. 
                # Let's use the property: tree_count[node] = _calc(right_child, left_child_max)
                # So the contribution of the right child is simply:
                # self.tree_count[node] - (count of elements in left child that are > left_child_max)
                # Actually, the correct recurrence is:
                # return self._calc(2 * node, start, mid, limit) + (self.tree_count[node] - self._calc(2 * node, start, mid, self.tree_max[2 * node]))
                # But since we are at the node, we already know the right child's 
                # contribution relative to the left child's max is stored.
                # Let's redefine: tree_count[node] is the number of visible buildings 
                # in the right subtree given the max of the left subtree.
                # Then the total visible in node is: _calc(left, limit) + (something).
                # Let's use a simpler approach for the logic.

    # Since the Segment Tree logic is getting complex to implement without errors,
    # let's use the property that we need to count j > i such that max(H[i+1...j-1]) < H[j].
    # This is equivalent to counting elements in the range [i+1, N-1] that are 
    # larger than all preceding elements in that range.
    
    # For a fixed i, we want the number of left-to-right maximums of H[i+1...N-1].
    # This can be solved by a Segment Tree where each node stores the max of its range
    # and a function `count(node, threshold)` that returns the number of L-to-R maximums
    # in that node's range that are greater than `threshold`.

    def build_tree(l, r):
        if l == r:
            return {'max': H[l], 'cnt': 1, 'l': l, 'r': r}
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        return {
            'max': max(left['max'], right['max']),
            'left': left,
            'right': right,
            'cnt': left['cnt'] + query_visible(right, left['max']),
            'l': l, 'r': r
        }

    def query_visible(node, threshold):
        if node['max'] <= threshold:
            return 0
        if node['l'] == node['r']:
            return 1 if node['max'] > threshold else 0
        
        # If left child's max is <= threshold, only right child matters
        if node['left']['max'] <= threshold:
            return query_visible(node['right'], threshold)
        else:
            # Left child contributes, and right child's contribution 
            # relative to left child's max is already pre-calculated.
            # Total = query_visible(left, threshold) + (node['cnt'] - left['cnt'])
            return query_visible(node['left'], threshold) + (node['cnt'] - node['left']['cnt'])

    # We need the answer for each i from 0 to N-1.
    # For a specific i, we need query_visible(root_of_range(i+1, N-1), -1).
    # Since we can't easily build a new tree for every i, we use the fact that
    # we only need the range [i+1, N-1]. We can use a Segment Tree and 
    # query the range [i+1, N-1].
    
    # However, the simplest way to implement this in Python without recursion 
    # limits and with O(N log^2 N) is to use a Segment Tree and a custom query.
    
    # Let's use a different approach: the answer for i is the number of 
    # L-to-R maximums in H[i+1...N-1].
    # This is a known problem. We can use a Segment Tree where each node 
    # stores the max of its range and the number of L-to-R maximums 
    # within its own range.
    
    # To avoid recursion, we can use a list-based segment tree.
    # But the `query_visible` is naturally recursive. 
    # Given N=2e5, we must increase recursion depth.
    sys.setrecursionlimit(300000)
    
    # Build the tree for the entire range [0, N-1]
    root = build_tree(0, N - 1)
    
    # For each i, we need the number of L-to-R maximums in [i+1, N-1].
    # We can't just use the root because the root is for [0, N-1].
    # We need a way to query the number of L-to-R maximums in a suffix.
    
    # A simpler observation: the number of j > i such that H[j] > max(H[i+1...j-1])
    # is exactly the number of elements in the suffix H[i+1...N-1] that are 
    # greater than all elements to their left in that suffix.
    
    # We can solve this by processing i from N-1 down to 0.
    # But the "L-to-R maximums of a suffix" is not easily updated.
    # Wait, the problem is simpler: for a fixed i, we count j > i such that 
    # H[j] is a prefix maximum of the array H[i+1...N-1].
    
    # Let's use the Segment Tree to find the answer for all i.
    # The number of L-to-R maximums in [i+1, N-1] can be found by:
    # 1. Find the first index k > i such that H[k] > H[i+1] (if i+1 < N).
    # 2. The answer is 1 + (L-to-R maximums in [k, N-1] given threshold H[k]).
    # This is still complex.
    
    # Correct approach: Use a Segment Tree to maintain the range [0, N-1].
    # To find L-to-R maximums in [i+1, N-1], we can query the range.
    # But the "threshold" depends on the maximum of the range to the left of the current block.
    
    # Let's use the property: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means j is a left-to-right maximum of the suffix starting at i+1.
    
    # Final attempt at logic: 
    # The number of such j is simply the number of elements in the 
    # "monotonic stack" of the suffix H[i+1...N-1] when processed from left to right.
    # Actually, the most efficient way is to use a Segment Tree and 
    # a function that counts visible elements in a range given a threshold.
    
    def get_ans(i):
        if i >= N - 1: return 0
        # We need L-to-R maximums in range [i+1, N-1]
        # We can use a helper that queries the segment tree for the range [i+1, N-1]
        def query_range(node, l, r, threshold):
            # This is tricky because the threshold changes.
            pass
            
    # Given the constraints and the problem type, the most reliable way 
    # to implement this is to use the build_tree logic but 
    # adapt it to query suffixes.
    # Since we need it for all i, we can use a Divide and Conquer approach.
    
    def solve_dc(l, r):
        # Returns a list of answers for i in [l, r]
        # This is also complex. 
        pass

    # Let's use the most straightforward O(N log^2 N) approach:
    # For each i, the answer is query_visible(root_of_range(i+1, N-1), -1).
    # To make this work, we can use a Segment Tree where we can 
    # query the number of visible elements in a range [L, R] given a threshold.
    
    # Since I must provide a working solution, I will use the 
    # recursive build and a modified query that handles the range [i+1, N-1].
    
    def query_suffix(node, l, r, threshold):
        # Count visible elements in range [l, r] given threshold
        if node['l'] > r or node['r'] < l:
            return 0, threshold
        if node['l'] >= l and node['r'] <= r:
            # Use the pre-calculated count, but it's relative to the left child's max.
            # We need to calculate it relative to the current threshold.
            res = query_visible(node, threshold)
            return res, max(threshold, node['max'])
        
        mid = (node