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
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to counting how many elements to the right of i 
    # are "right-to-left" maxima if we were looking from i onwards.
    # More simply: j satisfies the condition if H[j] is a new maximum 
    # encountered while scanning from index i+1 to N.
    
    # However, the problem asks for this for every i.
    # Let's rephrase: j is counted for i if for all k such that i < k < j, H[k] < H[j].
    # This means j is a "visible" building from i if we only care about 
    # buildings blocking the view based on the height of the target building j.
    
    # Correct interpretation:
    # For a fixed i, we count j > i such that max(H[i+1]...H[j-1]) < H[j].
    # Note: For j = i + 1, the range (i+1...j-1) is empty, so it always satisfies.
    # For j > i + 1, j satisfies if H[j] is greater than all heights between i and j.
    
    # This is a range query problem. For a fixed i, we want the number of j > i
    # such that H[j] > max_{i < k < j} H[k].
    # This is equivalent to counting elements in the sequence H[i+1...N] 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # We can solve this using a Segment Tree. 
    # Each node in the segment tree will store:
    # 1. The maximum height in its range.
    # 2. A value 'count' which is the number of visible buildings in that range
    #    given a certain height threshold from the left.
    
    # Since we need to do this for every i, we can use a recursive function 
    # that calculates how many elements in a range [L, R] are greater than 
    # a given height 'h', considering only those that are maxima within the range.
    
    tree_max = [0] * (4 * N)
    
    def build(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])

    def count_visible(node, start, end, threshold):
        # Returns the number of elements in [start, end] that are > threshold
        # and are maxima relative to elements to their left within [start, end].
        if tree_max[node] <= threshold:
            return 0
        if start == end:
            return 1 if tree_max[node] > threshold else 0
        
        mid = (start + end) // 2
        # If the max of the left child is <= threshold, 
        # nothing in the left child is visible, and the threshold for the 
        # right child remains the same.
        if tree_max[2 * node] <= threshold:
            return count_visible(2 * node + 1, mid + 1, end, threshold)
        
        # If the max of the left child > threshold, some elements in the left 
        # are visible. The right child's visibility depends on the max of the 
        # left child, NOT the original threshold.
        # The number of visible elements in the right child given the left child's 
        # max is (total_visible_in_node - visible_in_left_given_left_child_max).
        # Wait, the property is: 
        # f(node, threshold) = 
        #    if max(left) <= threshold: f(right, threshold)
        #    else: f(left, threshold) + (f(node, -inf) - f(left, -inf))
        # We precalculate f(node, -inf) as 'stored_count'.
        
        return count_visible(2 * node, start, mid, threshold) + (
            stored_count[node] - stored_count[2 * node]
        )

    # To avoid recursion depth and implement the logic above:
    # stored_count[node] = count_visible(2 * node + 1, mid + 1, end, tree_max[2 * node])
    # This is the number of elements in the right subtree that are visible 
    # given the maximum of the left subtree.
    
    # Since we cannot use loops or complex structures, we use a helper to 
    # initialize the stored_count using a recursive build.
    
    stored_count = [0] * (4 * N)
    
    def build_final(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            stored_count[node] = 1
            return
        mid = (start + end) // 2
        build_final(2 * node, start, mid)
        build_final(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        # Precompute how many in the right subtree are visible given the left subtree max
        stored_count[node] = count_visible_logic(2 * node + 1, mid + 1, end, tree_max[2 * node]) + 1 
        # The +1 is for the first element of the left subtree? No.
        # Let's redefine: stored_count[node] is the number of visible elements 
        # in the range [start, end] when threshold is -1.
        # For a leaf, it's 1.
        # For a node, it's count_visible(left, -1) + count_visible(right, max(left))
    
    # To avoid the function inside function issue with recursion and state,
    # we use a class or a custom recursive structure.
    
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.tree_max = [0] * (4 * self.n)
            self.stored_count = [0] * (4 * self.n)
            self.data = data
            self._build(1, 0, self.n - 1)

        def _count(self, node, start, end, threshold):
            if self.tree_max[node] <= threshold:
                return 0
            if start == end:
                return 1
            mid = (start + end) // 2
            if self.tree_max[2 * node] <= threshold:
                return self._count(2 * node + 1, mid + 1, end, threshold)
            return self._count(2 * node, start, mid, threshold) + (
                self.stored_count[node] - self.stored_count[2 * node]
            )

        def _build(self, node, start, end):
            if start == end:
                self.tree_max[node] = self.data[start]
                self.stored_count[node] = 1
                return
            mid = (start + end) // 2
            self._build(2 * node, start, mid)
            self._build(2 * node + 1, mid + 1, end)
            self.tree_max[node] = max(self.tree_max[2 * node], self.tree_max[2 * node + 1])
            # stored_count[node] = visible in left + visible in right given max of left
            self.stored_count[node] = self.stored_count[2 * node] + \
                                      self._count(2 * node + 1, mid + 1, end, self.tree_max[2 * node])

        def query(self, i):
            # We want visible elements in range [i+1, N-1]
            # Since we can't easily query ranges in this specific logic,
            # we can use the property that we only need suffixes.
            # However, the simplest way to get suffix [i+1, N-1] is to 
            # use a range query and handle the threshold.
            # But the problem asks for EVERY i.
            # A simpler approach: the number of j > i such that 
            # H[j] > max(H[i+1...j-1]) is exactly the number of 
            # right-to-left maxima of the suffix H[i+1...].
            # Wait, that's not right. It's left-to-right maxima of the suffix.
            pass

    # Let's use a different approach: 
    # For a fixed i, we count j \in [i+1, N-1] such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements that would remain if we 
    # processed the suffix H[i+1...N-1] and kept only those that are 
    # greater than all previous elements.
    
    # This is a classic problem solvable by a Segment Tree where each node 
    # stores the max and the count of visible elements.
    # To get the answer for all i, we can use the fact that we are 
    # querying suffixes. We can build the tree on the whole array and 
    # query the range [i+1, N-1].
    
    # Since we need a range query [L, R], the _count function needs to be 
    # wrapped in a range query logic.
    
    def range_query(node, start, end, L, R, threshold):
        # This is tricky because the threshold changes.
        # We can use a helper that returns (count, max_val)
        pass

# The above is getting complex. Let's use the property:
# j is counted for i if H[j] > max(H[i+1...j-1]).
# This means for a fixed j, it is counted for all i < j-1 such that 
# max(H[i+1...j-1]) < H[j].
# This is equivalent to: i+1 > (index of the first element to the left of j 
# that is > H[j]).
# Let L[j] be the index of the first element to the left of j such that H[L[j]] > H[j].
# If no such element exists, L[j] = -1.
# Then j is counted for all i such that L[j] <= i < j-1.
# Wait, the condition is: no building taller than Building j BETWEEN i and j.
# So for i and j, we check k \in \{i+1, \dots, j-1\}.
# If i = j-1, the set is empty, always true.
# If i < j-1, we need H[k] < H[j] for all k \in \{i+1, \dots, j-1\}.
# This means the maximum height in the range [i+1, j-1] must be < H[j].
# This is true if and only if the first index k < j where H[k] > H[j] 
# is $\le i$.
# Let prev_greater[j] = max({k | k < j and H[k] > H[j]} \cup \{-1\}).
# The condition is: prev_greater[j] $\le i$.
# Also we need $i < j$.
# So for a fixed $i$, we count $j > i$ such that prev_greater[j] $\le i$.
# Note: $j$ must be $> i$.
# For a fixed $i$, we want $\sum_{j=i+1}^{N-1} [prev\_greater[j] \le i]$.
# This is a 2D range counting problem: count $(j, prev\_greater[j])$ 
# such that $j \in [i+1, N-1]$ and $prev\_greater[j] \in [-1, i]$.
# Since we can process $i$ from $N-1$ down to 0, we can use a Fenwick tree.
# As $i$ decreases, the condition $