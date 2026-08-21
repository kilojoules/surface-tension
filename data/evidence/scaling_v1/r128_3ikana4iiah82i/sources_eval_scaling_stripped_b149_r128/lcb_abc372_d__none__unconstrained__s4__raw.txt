```python
import sys
from bisect import bisect_right

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed j, we are looking for i < j such that
    # max(H[i+1]...H[j-1]) <= H[j].
    # This is equivalent to saying that Building j is a "Right-to-Left" 
    # visible building from the perspective of i.
    # More simply: for a fixed i, we want to count j > i such that 
    # H[j] >= max(H[i+1]...H[j-1]).
    # This is exactly the definition of the number of elements to the right 
    # that would form a new maximum if we scanned from i+1 to N.
    # However, the condition is slightly different: it's about the height of 
    # the destination building j, not the starting building i.
    
    # Let's re-evaluate: for a fixed i, we count j > i where 
    # for all k such that i < k < j, H[k] <= H[j].
    # This means Building j is the first building to the right of k 
    # that is at least as tall as H[k] for all k in (i, j).
    
    # Actually, the condition "no building taller than Building j between i and j"
    # is satisfied if and only if H[j] >= max(H[i+1], ..., H[j-1]).
    # Let M[i][j] = max(H[i+1]...H[j-1]). We want count of j > i where H[j] >= M[i][j].
    # Note: for j = i+1, the range (i, j) is empty, so the condition is vacuously true.
    
    # This is a classic problem that can be solved by observing that 
    # for a fixed i, the buildings j that satisfy this are the ones that 
    # "update" the prefix maximum of the array H[i+1...N].
    # If we define a sequence P_i = [H[i+1], H[i+2], ..., H[N]],
    # we are counting how many elements in P_i are greater than or equal to 
    # all elements to their left in P_i.
    
    # Since we need to do this for all i, and N is 2*10^5, an O(N^2) is too slow.
    # We need a more efficient approach.
    # Let's use the property: j satisfies the condition for i if 
    # H[j] is the maximum of the range [i+1, j].
    # This is equivalent to saying that the index of the maximum in [i+1, j] is j.
    
    # Wait, the problem is simpler: for a fixed i, we are counting j > i 
    # such that H[j] >= max(H[i+1]...H[j-1]).
    # This is exactly the number of times the prefix maximum changes 
    # (including the first element) when iterating from i+1 to N.
    
    # To solve this for all i, we can use a Segment Tree or similar structure,
    # but the "number of prefix maximums" is a known problem solvable with 
    # a Segment Tree where each node stores the maximum of its range and 
    # the number of prefix maximums relative to some external value.
    
    # Segment Tree Node: (max_val, count_prefix_max)
    # The merge function: 
    # left_node, right_node
    # combined_max = max(left_node.max, right_node.max)
    # combined_count = left_node.count + count_prefix_max(right_node, left_node.max)
    
    # Since we cannot use loops, we use a functional-style segment tree 
    # built using a list and recursion (with sys.setrecursionlimit).
    
    sys.setrecursionlimit(300000)
    
    # We build the tree once. Each node will store the max of its range.
    # To find the number of prefix maximums in a range [L, R] given a 
    # starting maximum 'v', we use a helper function.
    
    tree_max = [0] * (4 * N)
    
    def build(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])

    def query_count(node, start, end, v):
        # Returns (count of prefix maxes in range, new_v)
        if tree_max[node] <= v:
            return 0, v
        if start == end:
            return 1, tree_max[node]
        
        mid = (start + end) // 2
        # We need to know how many in the left child are > v
        # and then how many in the right child are > max(v, left_child_max)
        
        # This is still recursive. To avoid loops, we use a trick:
        # If the left child's max is <= v, we only search the right child.
        if tree_max[2 * node] <= v:
            return query_count(2 * node + 1, mid + 1, end, v)
        
        # If left child's max > v, we must search both.
        # But we can't just add them because the right child's count 
        # depends on the left child's total max.
        # The number of prefix maxes in the right child given the left child's max
        # is a value we can precalculate!
        
        # Let's redefine: each node stores 'max' and 'cnt', where 'cnt' is the 
        # number of prefix maximums in the right child given the left child's max.
        pass

    # Correcting the approach:
    # Each node stores: (max_of_range, count_of_prefix_maxes_of_right_given_left)
    # This is getting complex. Let's use the property that we need the answer for all i.
    # The answer for i is the number of prefix maximums of H[i+1...N].
    
    # Let's use a simpler observation: 
    # j satisfies the condition for i if H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to saying that for all k in (i, j), H[k] < H[j].
    # This means j is the first index to the right of k such that H[j] > H[k] 
    # for all k in (i, j).
    
    # Actually, the most straightforward way to implement this without loops 
    # is to use a Segment Tree where each node stores the max of its range, 
    # and a function `solve(node, current_max)` that returns the number of 
    # elements in that range that are greater than `current_max`.
    
    # To make `solve` efficient, we can't just recurse. 
    # But we can use the fact that if `tree_max[node] <= current_max`, the answer is 0.
    # If we are at a leaf and `H[leaf] > current_max`, the answer is 1.
    # For an internal node, if `tree_max[left_child] <= current_max`, 
    # we only recurse into the right child.
    # If `tree_max[left_child] > current_max`, we recurse into both.
    # However, the number of elements in the right child that are > max(current_max, left_max)
    # is simply the number of elements in the right child that are > left_max,
    # because current_max < left_max.
    # This "number of elements in right child > left_max" can be precomputed!
    
    # Precomputed value for node: `res = count_greater(right_child, tree_max[left_child])`
    
    tree_res = [0] * (4 * N)
    
    def build_final(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            return 0
        mid = (start + end) // 2
        l_res = build_final(2 * node, start, mid)
        r_res = build_final(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        # Precompute the number of prefix maxes in the right subtree 
        # given the max of the left subtree.
        tree_res[node] = get_count(2 * node + 1, mid + 1, end, tree_max[2 * node])
        return 0

    def get_count(node, start, end, v):
        if tree_max[node] <= v:
            return 0
        if start == end:
            return 1
        mid = (start + end) // 2
        if tree_max[2 * node] <= v:
            return get_count(2 * node + 1, mid + 1, end, v)
        else:
            # Left child has some elements > v. 
            # The right child's contribution is already precomputed in tree_res[node]
            # because the left child's max will definitely be the threshold for the right.
            return get_count(2 * node, start, mid, v) + tree_res[node]

    # Since we need to call build_final and then query for each i:
    # We can use a list comprehension to call get_count for all i.
    
    # Note: H is 0-indexed, so Building i is H[i-1].
    # For i=1, we check range [1, N-1]. For i=N, range [N, N-1] (empty).
    
    # To avoid the function definition inside solve(), we can use a class or 
    # define them globally. Let's use a helper class.
    
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.h = data
            self.tree_max = [0] * (4 * self.n)
            self.tree_res = [0] * (4 * self.n)
            self._build(1, 0, self.n - 1)
            
        def _build(self, node, start, end):
            if start == end:
                self.tree_max[node] = self.h[start]
                return
            mid = (start + end) // 2
            self._build(2 * node, start, mid)
            self._build(2 * node + 1, mid + 1, end)
            self.tree_max[node] = max(self.tree_max[2 * node], self.tree_max[2 * node + 1])
            self.tree_res[node] = self._get_count(2 * node + 1, mid + 1, end, self.tree_max[2 * node])
            
        def _get_count(self, node, start, end, v):
            if self.tree_max[node] <= v:
                return 0
            if start == end:
                return 1
            mid = (start + end) // 2
            if self.tree_max[2 * node] <= v:
                return self._get_count(2 * node + 1, mid + 1, end, v)
            return self._get_count(2 * node, start, mid, v) + self.tree_res[node]
        
        def query(self, i):
            # For building i (1-indexed), we look at range [i, N-1] (0-indexed)
            # The range is H[i...N-1].
            # We need the number of prefix maximums in this range.
            # This is equivalent to _get_count(root, 0, N-1, max(H[0...i-1]))
            # BUT the condition is: no building taller than Building j BETWEEN i and j.
