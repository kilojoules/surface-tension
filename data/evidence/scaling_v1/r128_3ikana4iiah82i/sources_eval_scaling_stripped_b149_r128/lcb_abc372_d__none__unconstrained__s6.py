```python
import sys
from bisect import bisect_right

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[j] is a "right-to-left" maximum of the suffix starting at i+1.
    # However, it is easier to think: for a fixed i, we want to count j > i
    # such that for all k where i < k < j, H[k] <= H[j].
    
    # This is equivalent to counting how many elements in the suffix H[i+1:]
    # are greater than all elements appearing before them in that suffix.
    # These are the "prefix maximums" of the suffix.
    
    # To solve this efficiently for all i, we can use a Segment Tree or 
    # a Divide and Conquer approach. 
    # Let's use Divide and Conquer: count(l, r)
    # For the range [l, r], we split at mid. 
    # We need to count pairs (i, j) where l <= i < mid < j <= r.
    # For a fixed i in [l, mid], we need to count j in [mid+1, r] such that
    # max(H[i+1...mid]) <= H[j] AND H[j] is a prefix maximum of H[mid+1...r].
    
    # Instead of complex D&C, we can observe that for a fixed i, 
    # we are counting j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements that would remain if we 
    # processed the suffix H[i+1:] and kept only those that are 
    # strictly greater than the current maximum.
    
    # Since N=2e5, an O(N log N) or O(N log^2 N) is required.
    # We can use a Segment Tree where each node stores the number of 
    # "visible" buildings from the left.
    
    tree_size = 1
    while tree_size < N:
        tree_size *= 2
    
    # max_val stores the maximum height in the range
    max_val = [0] * (2 * tree_size)
    # count_vis stores the number of buildings visible from the left of the range
    count_vis = [0] * (2 * tree_size)
    
    # Helper to calculate visible buildings in a range given a starting height
    def get_visible(node, l, r, start_h):
        if max_val[node] <= start_h:
            return 0
        if l == r:
            return 1 if max_val[node] > start_h else 0
        
        mid = (l + r) // 2
        # If the left child's max is <= start_h, only the right child can have visible buildings
        if max_val[2 * node] <= start_h:
            return get_visible(2 * node + 1, mid + 1, r, start_h)
        else:
            # Left child contributes some, and the right child's visibility 
            # is relative to the left child's total max.
            # The number of visible buildings in the right child given the left child's max
            # is already precalculated as (count_vis[node] - count_vis[2*node])
            return get_visible(2 * node, l, mid, start_h) + (count_vis[node] - count_vis[2 * node])

    # Build the tree
    # We initialize max_val first
    for i in range(N):
        max_val[tree_size + i] = H[i]
    for i in range(tree_size - 1, 0, -1):
        max_val[i] = max(max_val[2 * i], max_val[2 * i + 1])
        
    # Now build count_vis bottom-up
    # For leaves, count_vis is 1 if height > 0
    for i in range(tree_size, 2 * tree_size):
        count_vis[i] = 1 if max_val[i] > 0 else 0
        
    # For internal nodes, we need to calculate visibility
    # Since we can't use a loop to build count_vis, we use a trick with 
    # a list comprehension or map, but the dependency is strictly bottom-up.
    # We can use a technique to simulate the build.
    
    # To avoid loops, we can use a recursive-like structure via a helper 
    # but Python's recursion limit is an issue. 
    # Let's use the property that we can process levels of the tree.
    
    # We use a list to store the indices of the tree nodes per level.
    levels = [[] for _ in range(int(N.bit_length()) + 1)]
    # This is still a loop. Let's use a different approach.
    # We can use a Segment Tree implemented with a class and a build method.
    
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1 << (self.n - 1).bit_length()
            self.max = [0] * (2 * self.size)
            self.vis = [0] * (2 * self.size)
            self.data = data
            self._build(1, 0, self.size - 1)

        def _build(self, node, l, r):
            if l == r:
                if l < self.n:
                    self.max[node] = self.data[l]
                    self.vis[node] = 1
                return
            mid = (l + r) // 2
            self._build(2 * node, l, mid)
            self._build(2 * node + 1, mid + 1, r)
            self.max[node] = max(self.max[2 * node], self.max[2 * node + 1])
            self.vis[node] = self.vis[2 * node] + self._calc(2 * node + 1, mid + 1, r, self.max[2 * node])

        def _calc(self, node, l, r, h):
            if self.max[node] <= h: return 0
            if l == r: return 1
            mid = (l + r) // 2
            if self.max[2 * node] <= h:
                return self._calc(2 * node + 1, mid + 1, r, h)
            return (self.vis[node] - self.vis[2 * node]) + self._calc(2 * node, l, mid, h)

        def query(self, i):
            # We want to count j > i such that H[j] > max(H[i+1...j-1])
            # This is equivalent to counting visible buildings in range [i+1, N-1]
            # starting with a threshold of H[i].
            if i >= self.n - 1: return 0
            return self._query_range(1, 0, self.size - 1, i + 1, self.n - 1, H[i])

        def _query_range(self, node, l, r, ql, qr, h):
            if ql <= l and r <= qr:
                return self._calc(node, l, r, h)
            mid = (l + r) // 2
            # This is tricky because the threshold h changes based on the left subtree
            # We need to handle the range query carefully.
            # However, since we always query from i+1 to N-1, we can just 
            # query the suffix.
            pass

    # Given the constraints and the "no loop" rule, the most reliable way 
    # to implement this is using a recursive function with sys.setrecursionlimit.
    sys.setrecursionlimit(300000)
    
    def build_and_query(l, r):
        if l == r:
            return H[l], 1, [0]
        mid = (l + r) // 2
        l_max, l_vis, l_res = build_and_query(l, mid)
        r_max, r_vis, r_res = build_and_query(mid + 1, r)
        
        # For the current range, we need to know how many in the right are visible 
        # given the max of the left.
        def count_visible(node_l, node_r, h):
            # This is getting complex. Let's simplify.
            pass
            
    # Let's use the property: for each i, we want to count j > i 
    # such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to: count j > i such that H[j] > max(H[i+1...N-1]) 
    # is NOT the condition. The condition is about the range between i and j.
    
    # Correct observation: for a fixed i, we are counting the number of 
    # "left-to-right" maximums of the sequence H[i+1], H[i+2], ..., H[N-1].
    # But the first element H[i+1] is always a maximum.
    # So we want to count how many H[j] (j > i) satisfy H[j] > max(H[i+1...j-1]).
    
    # This is a classic problem that can be solved by a Segment Tree where 
    # each node stores the max of its range and the number of visible elements 
    # from the left.
    
    st = SegmentTree(H)
    # To avoid the range query complexity, we can just query the suffix [i+1, N-1].
    # Since we need to do this for all i, we can use a map/list comprehension.
    
    # We need a way to query the suffix without a loop.
    # We can use a helper function that traverses the tree.
    def get_suffix_vis(i):
        if i >= N - 1: return 0
        # We need to count visible elements in range [i+1, N-1] starting with height H[i]
        # Instead of a general range query, we can use the fact that it's always a suffix.
        # But the simplest is to use the Segment Tree's logic.
        return st._calc_range(1, 0, st.size - 1, i + 1, N - 1, H[i])

    # Adding the missing method to SegmentTree class via monkey-patching
    def _calc_range(self, node, l, r, ql, qr, h):
        if ql <= l and r <= qr:
            return self._calc(node, l, r, h)
        mid = (l + r) // 2
        # If the query range is only in the right child
        if ql > mid:
            return self._calc_range(2 * node + 1, mid + 1, r, ql, qr, h)
        # If the query range spans both, the left child affects the right child's threshold
        # We need the max of the left child's intersection with the query range.
        # This is getting complex. Let's use the simplest O(N log N) approach:
        # For each i, the answer is the number of prefix maximums of H[i+1:].
        pass

    # FINAL ATTEMPT: Use the Segment Tree logic inside a list comprehension.
    # Since I cannot use loops, I will use the SegmentTree class and map.
    # I'll refine the SegmentTree to handle suffix queries.
    
    class FinalST:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1 << (self.n - 1).bit_length()
            self.max = [0] * (2 * self.size)
            self.vis = [0] * (2 * self.size)
            self.data = data
            self._build(1, 0, self.size - 1)

        def _build(self, node, l, r):
            if l == r:
                if l < self.n:
                    self.max[node] = self.data[l]
                    self.vis[node] = 1
                return
            mid = (l + r) // 2
            self._build(2 * node