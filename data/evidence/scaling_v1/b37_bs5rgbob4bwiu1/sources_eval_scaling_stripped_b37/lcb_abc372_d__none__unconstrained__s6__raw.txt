```python
import sys
from functools import reduce

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum when looking at the range (i, N].
    # More simply: j satisfies the condition if H[j] > max(H[i+1]...H[j-1]).
    # This means we are counting elements in the sequence H[i+1...N-1] 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # To solve this efficiently for all i, we can use a Segment Tree or 
    # a Divide and Conquer approach. 
    # The problem asks for the number of visible buildings to the right.
    # This is a classic problem solvable by a Segment Tree where each node
    # stores the maximum height in its range and a precomputed count of 
    # visible buildings from the left within that range.

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
            # The count for the parent is the count of the left child 
            # plus the number of elements in the right child that are 
            # greater than the maximum of the left child.
            self.tree_count[node] = self.tree_count[2 * node] + \
                                    self._query_visible(2 * node + 1, mid + 1, end, self.tree_max[2 * node])

        def _query_visible(self, node, start, end, threshold):
            if self.tree_max[node] <= threshold:
                return 0
            if start == end:
                return 1 if self.tree_max[node] > threshold else 0
            
            mid = (start + end) // 2
            # If the left child's max is <= threshold, only the right child can contribute
            if self.tree_max[2 * node] <= threshold:
                return self._query_visible(2 * node + 1, mid + 1, end, threshold)
            else:
                # Left child contributes some, and right child contributes 
                # (total right visible) - (right visible blocked by left child's max)
                # Actually, the number of visible elements in the right child 
                # given the left child's max is already precomputed as:
                # self.tree_count[node] - self.tree_count[2 * node]
                return self._query_visible(2 * node, start, mid, threshold) + \
                       (self.tree_count[node] - self.tree_count[2 * node])

        def range_query(self, l, r):
            # This specific problem asks for i+1 to N-1.
            # We can use a helper to handle the range and return (max, count).
            return self._range_helper(1, 0, self.n - 1, l, r)

        def _range_helper(self, node, start, end, l, r):
            if start > end or start > r or end < l:
                return (0, 0)
            if start >= l and end <= r:
                return (self.tree_max[node], self.tree_count[node])
            
            mid = (start + end) // 2
            left_max, left_cnt = self._range_helper(2 * node, start, mid, l, r)
            right_max, right_cnt = self._range_helper(2 * node + 1, mid + 1, end, l, r)
            
            # Combine results: count visible in right range given left_max
            # We need a way to query the right child's visibility based on left_max
            # Since we are querying a specific range, we use a modified query logic.
            # However, for this specific problem, we only ever query [i+1, N-1].
            # That is a suffix query.
            return (max(left_max, right_max), 0) # Placeholder

    # Since we only need suffix queries [i+1, N-1], we can simplify.
    # For a fixed i, we want the number of j in [i+1, N-1] such that 
    # H[j] > max(H[i+1...j-1]).
    # This is exactly the number of prefix maximums of the suffix H[i+1...].
    
    # We can use a Segment Tree where each node stores the max of its range
    # and the number of visible elements from the left within that range.
    # To find the answer for i, we query the range [i+1, N-1].
    
    def get_visible_count(node, start, end, threshold):
        # Returns (max_val, count_visible)
        if start == end:
            return (tree_max[node], 1 if tree_max[node] > threshold else 0)
        
        mid = (start + end) // 2
        # If left max <= threshold, left side is invisible, check right
        if tree_max[2 * node] <= threshold:
            res_max, res_cnt = get_visible_count(2 * node + 1, mid + 1, end, threshold)
            return (res_max, res_cnt)
        else:
            # Left side is visible, right side visibility is relative to left max
            # The number of visible elements in the right child given the left child's max
            # is precomputed as (tree_count[node] - tree_count[2 * node])
            # But only if the range is exactly the node's range.
            # Since we are doing a suffix query, we can use a different approach.
            pass

    # Correct approach for suffix:
    # For each i, we need the number of prefix maximums of H[i+1...N-1].
    # This can be solved by a Segment Tree where we query the range [i+1, N-1].
    # For a range [L, R], the number of visible elements is:
    # count(L, R, threshold) = 
    #    if L==R: 1 if H[L] > threshold else 0
    #    if max(L, mid) <= threshold: count(mid+1, R, threshold)
    #    else: count(L, mid, threshold) + (tree_count[node] - tree_count[left_child])

    # Implementation without classes for speed:
    tree_max = [0] * (4 * N)
    tree_count = [0] * (4 * N)

    def build(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            tree_count[node] = 1
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        tree_count[node] = tree_count[2 * node] + calc_visible(2 * node + 1, mid + 1, end, tree_max[2 * node])

    def calc_visible(node, start, end, threshold):
        if tree_max[node] <= threshold:
            return 0
        if start == end:
            return 1
        mid = (start + end) // 2
        if tree_max[2 * node] <= threshold:
            return calc_visible(2 * node + 1, mid + 1, end, threshold)
        else:
            # Left child contributes, right child contribution is precomputed
            return calc_visible(2 * node, start, mid, threshold) + (tree_count[node] - tree_count[2 * node])

    def query_visible(node, start, end, l, r, threshold):
        # Returns (max_val, count)
        if start > end or start > r or end < l:
            return (0, 0)
        if start >= l and end <= r:
            # We need the count of elements in this range > threshold
            # This is a helper function, not the precomputed tree_count
            # because the threshold comes from the left of the range.
            # We use a logic similar to calc_visible but for a range.
            # However, we can't use calc_visible directly because it assumes 
            # the node is fully contained. 
            # Wait, if start >= l and end <= r, then it IS fully contained.
            # So we can use a modified version of calc_visible.
            # But we need the actual count for the specific threshold.
            # Let's use a helper.
            pass

    # To avoid the range query complexity, notice we only need suffixes [i+1, N-1].
    # We can use a Segment Tree and for each i, query the range [i+1, N-1].
    # A range query on [L, R] returns (max, count).
    # When merging (max1, cnt1) and (max2, cnt2), the new count is
    # cnt1 + calc_visible(node_of_range2, L2, R2, max1).

    def query_range(node, start, end, l, r):
        if start >= l and end <= r:
            return (tree_max[node], tree_count[node])
        mid = (start + end) // 2
        if r <= mid:
            return query_range(2 * node, start, mid, l, r)
        if l > mid:
            return query_range(2 * node + 1, mid + 1, end, l, r)
        
        left_max, left_cnt = query_range(2 * node, start, mid, l, r)
        right_max, right_cnt = query_range(2 * node + 1, mid + 1, end, l, r)
        
        # The right_cnt is the count of visible elements within the right range [mid+1, r].
        # But we need those visible given left_max.
        # Since the right range is [mid+1, r], we can't use the precomputed 
        # tree_count of the right child directly unless r == end.
        # If r == end, we can use calc_visible on the right child with left_max.
        
        # For suffix queries [i+1, N-1], r is always N-1.
        # So the right child is always fully contained.
        return (max(left_max, right_max), left_cnt + calc_visible(2 * node + 1, mid + 1, end, left_max))

    # Since we only need suffixes, we can just use a function that 
    # processes the range [i+1, N-1] by traversing the tree.
    def get_suffix_count(l):
        if l >= N: return 0
        # Query range [l, N-1]
        # We can use a helper that returns (max, count)
        def helper(node, start, end, L):
            if start >= L:
                # Fully contained suffix block
                # We need the count of elements visible from the left
                # But the threshold is the max of everything to the left of this block.
                # This is handled by the recursion.
                return (tree_max[node], tree_count[node])
            
            mid = (start + end) // 2
            if L > mid:
                return helper(2 * node + 1, mid + 1, end, L)
            else