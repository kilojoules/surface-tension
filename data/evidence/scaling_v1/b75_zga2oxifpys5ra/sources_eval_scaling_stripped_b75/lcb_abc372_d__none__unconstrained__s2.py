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
    # such that H[j] is a new maximum encountered while scanning from i+1 to N.
    # However, the condition is specifically about buildings BETWEEN i and j.
    # Let's re-read: "no building taller than Building j between Buildings i and j".
    # This means for all k where i < k < j, H[k] <= H[j].
    # This is equivalent to saying that Building j is visible from Building i
    # if we only care about the height of the target building j.
    # Actually, the standard interpretation of this specific problem (from ABC)
    # is that j satisfies the condition if H[j] > max(H[i+1]...H[j-1]).
    
    # To solve this efficiently for all i, we can use a Segment Tree or a similar
    # structure, but since we need to do this for every i, a naive approach is O(N^2).
    # With N=2*10^5, we need O(N log N) or O(N).
    # This is a classic problem that can be solved by processing the buildings
    # and using a data structure to count elements.
    # But wait, the condition is: for a fixed i, count j > i such that 
    # for all k in (i, j), H[k] < H[j].
    # This is equivalent to saying that j is a "right-to-left" maximum 
    # if we were looking from j backwards to i+1.
    
    # Correct approach:
    # For a fixed i, the indices j that satisfy the condition are those where
    # H[j] > max(H[i+1], ..., H[j-1]).
    # This looks like we are counting how many times the prefix maximum changes
    # when scanning from i+1 to N.
    
    # This is a known problem that can be solved using a Segment Tree where each
    # node stores the maximum value in its range and the number of visible 
    # buildings from the left.
    
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.tree_max = [0] * (4 * self.n)
            self.tree_cnt = [0] * (4 * self.n)
            self._build(data, 1, 0, self.n - 1)

        def _build(self, data, node, start, end):
            if start == end:
                self.tree_max[node] = data[start]
                self.tree_cnt[node] = 1
                return
            mid = (start + end) // 2
            self._build(data, 2 * node, start, mid)
            self._build(data, 2 * node + 1, mid + 1, end)
            self.tree_max[node] = max(self.tree_max[2 * node], self.tree_max[2 * node + 1])
            self.tree_cnt[node] = self._calc(2 * node + 1, mid + 1, end, self.tree_max[2 * node])

        def _calc(self, node, start, end, limit):
            if self.tree_max[node] <= limit:
                return 0
            if start == end:
                return 1
            mid = (start + end) // 2
            if self.tree_max[2 * node] <= limit:
                return self._calc(2 * node + 1, mid + 1, end, limit)
            else:
                # The left child contributes some, and the right child's 
                # contribution is already pre-calculated relative to the left child's max.
                return (self.tree_cnt[node] - self.tree_cnt[2 * node]) + \
                       self._calc(2 * node, start, mid, limit)

        def query(self, l, r):
            # This specific problem requires counting visible elements in range [i+1, N-1]
            # We can use a helper to traverse the tree.
            return self._query_recursive(1, 0, self.n - 1, l, r, -1)

        def _query_recursive(self, node, start, end, l, r, current_max):
            if start > r or end < l:
                return 0, current_max
            if start >= l and end <= r:
                count = self._calc_external(node, start, end, current_max)
                return count, max(current_max, self.tree_max[node])
            
            mid = (start + end) // 2
            left_count, left_max = self._query_recursive(2 * node, start, mid, l, r, current_max)
            right_count, right_max = self._query_recursive(2 * node + 1, mid + 1, end, l, r, left_max)
            return left_count + right_count, right_max

        def _calc_external(self, node, start, end, limit):
            if self.tree_max[node] <= limit:
                return 0
            if start == end:
                return 1
            mid = (start + end) // 2
            if self.tree_max[2 * node] <= limit:
                return self._calc_external(2 * node + 1, mid + 1, end, limit)
            else:
                return (self.tree_cnt[node] - self.tree_cnt[2 * node]) + \
                       self._calc_external(2 * node, start, mid, limit)

    # Since we need to query for every i from 0 to N-1:
    # For i, we query the range [i+1, N-1].
    # The Segment Tree approach above is O(N log^2 N).
    
    # Implementation of the logic without a class for speed in Python:
    def build_and_solve():
        # Using a list-based segment tree to avoid recursion limits and overhead
        # But the logic requires a descent. Let's use a simpler approach.
        # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
        # This is exactly the number of elements in the upper hull/monotonic chain.
        # Actually, the most efficient way to solve this for all i is to use 
        # the property that we are counting prefix maximums of the suffix.
        
        # We can use a Divide and Conquer approach or a Segment Tree.
        # Given the constraints and Python, we must be careful.
        # Let's use the Segment Tree logic but optimized.
        pass

    # Re-evaluating: the problem is equivalent to:
    # For each i, count j in [i+1, N-1] such that H[j] > max(H[i+1...j-1]).
    # This is simply the number of left-to-right maxima in the subarray H[i+1...N-1].
    
    # Using a Segment Tree where each node stores:
    # - max_val: maximum height in the range
    # - count: number of visible buildings from the left of this range
    
    # To avoid recursion, we can use a technique to simulate it or use a 
    # different approach. However, N=2*10^5 is large for Python.
    # Let's use the property that we can solve this by processing queries.
    # But the queries are not independent.
    
    # Final attempt at a performant structure:
    # We can use a Fenwick tree if we process heights in a certain order, 
    # but that's for different problems. 
    # The most reliable way is the Segment Tree with the 'calc' function.
    
    # To make it pass in Python, we avoid the class and use functions.
    # We use a list for the tree and a stack-based approach for the recursive calls.
    
    # Since I must provide a complete working solution, I will implement the 
    # Segment Tree logic carefully.
    
    # Actually, there is a simpler O(N log N) approach using a Segment Tree 
    # and the fact that we are querying suffixes.
    # For a range [L, R], the number of visible elements from the left is:
    # visible(L, mid) + visible_with_limit(mid+1, R, max(L, mid))
    
    # Let's implement the Segment Tree logic without a class.
    
    # Using a helper to handle the 'calc' logic without deep recursion
    def get_visible(node, start, end, limit, tree_max, tree_cnt):
        # Use a stack to simulate recursion for the calc function
        # stack stores (node, start, end, limit)
        # This is tricky because we need the return value.
        # Instead, let's use the recursive version and increase recursion depth.
        sys.setrecursionlimit(300000)
        
        def _calc(n, s, e, lim):
            if tree_max[n] <= lim: return 0
            if s == e: return 1
            m = (s + e) // 2
            if tree_max[2*n] <= lim:
                return _calc(2*n+1, m+1, e, lim)
            return (tree_cnt[n] - tree_cnt[2*n]) + _calc(2*n, s, m, lim)
        
        return _calc

    # Build the tree
    def build(n, s, e, H, tree_max, tree_cnt, _calc_func):
        if s == e:
            tree_max[n] = H[s]
            tree_cnt[n] = 1
            return
        m = (s + e) // 2
        build(2*n, s, m, H, tree_max, tree_cnt, _calc_func)
        build(2*n+1, m+1, e, H, tree_max, tree_cnt, _calc_func)
        tree_max[n] = max(tree_max[2*n], tree_max[2*n+1])
        tree_cnt[n] = _calc_func(2*n+1, m+1, e, tree_max[2*n])

    # Query the tree
    def query(n, s, e, l, r, current_max, tree_max, tree_cnt, _calc_func):
        if s > r or e < l:
            return 0, current_max
        if s >= l and e <= r:
            res = _calc_func(n, s, e, current_max)
            return res, max(current_max, tree_max[n])
        m = (s + e) // 2
        left_res, left_max = query(2*n, s, m, l, r, current_max, tree_max, tree_cnt, _calc_func)
        right_res, right_max = query(2*n+1, m+1, e, l, r, left_max, tree_max, tree_cnt, _calc_func)
        return left_res + right_res, right_max

    # To avoid the closure/recursion issue in the build, we define _calc first.
    # But _calc needs tree_max and tree_cnt. We can use a list to hold them.
    
    # Final structure:
    # 1. Set recursion limit.
    # 2. Define a function that contains the logic.
    # 3. Use a list for the tree.
    
    # Since we cannot use a class and need to avoid recursion depth issues,
    # we can use a list-based segment tree and a manual stack for the query.
    # However, the 'calc' function is the core.
    
    # Let's use a simpler approach: for each i, we want to count j > i 
    # such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting elements in the set of 
    # "right-side" maxima.
    
    # Given the constraints and Python, the most efficient way is to 
    # use the Segment Tree and `sys.setrecursionlimit`.

    sys.setrecursionlimit(400000)
    
    # We wrap everything in a function to avoid global scope and allow 
    # the recursive functions to access the lists.
    def run():
        # Re