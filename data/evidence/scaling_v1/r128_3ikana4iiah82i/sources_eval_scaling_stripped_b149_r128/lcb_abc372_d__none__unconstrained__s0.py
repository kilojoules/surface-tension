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
    # This is equivalent to saying that Building j is a "right-to-left" 
    # visible building from the perspective of i.
    # More simply: for a fixed i, we want to count j > i such that
    # H[j] >= max(H[i+1]...H[j-1]).
    # This is exactly the definition of the number of elements in the 
    # "upper envelope" or the number of times a new maximum is encountered
    # when scanning from i+1 to N.
    # However, the condition is specifically about buildings BETWEEN i and j.
    # Let's re-evaluate: j satisfies the condition if for all k: i < k < j, H[k] <= H[j].
    # This means H[j] must be >= the maximum of the range (i, j).
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many j > i are "visible".
    # A building j is visible from i if it is the first building of its height 
    # (or taller) encountered.
    # Actually, the simplest interpretation: 
    # For a fixed i, we are counting j > i such that H[j] is a 
    # "running maximum" of the sequence H[i+1], H[i+2], ..., H[N].
    # Wait, the condition is: no building taller than H[j] between i and j.
    # That means H[k] <= H[j] for all i < k < j.
    # This is true if and only if H[j] >= max(H[i+1], ..., H[j-1]).
    # This is exactly the definition of a prefix maximum of the suffix starting at i+1.
    # The number of such j is the number of distinct values in the 
    # sequence of prefix maximums of H[i+1...N].
    # Since all H_i are distinct, this is the number of elements that 
    # set a new maximum as we scan from i+1 to N.
    
    # To solve this for all i in O(N log N) or O(N):
    # We can use a Segment Tree or a similar structure, but since we need 
    # the count of prefix maximums for every suffix, this is a classic 
    # problem solvable by a Segment Tree where each node stores the 
    # maximum of its range and a precomputed count of prefix maximums 
    # relative to a given height.
    
    # However, given the constraints and the nature of the problem, 
    # we can use the property that we are counting j > i such that 
    # H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to counting how many j > i are NOT "blocked" 
    # by some k (i < k < j) where H[k] > H[j].
    
    # Let's use the Segment Tree approach.
    # Each node in the Segment Tree will store:
    # 1. The maximum value in its range.
    # 2. A function/method to count how many elements in its range 
    #    are prefix maximums given a starting maximum from the left.
    
    # Since we cannot use loops, we use recursion for the Segment Tree.
    # But Python's recursion limit and speed might be an issue.
    # Let's implement the Segment Tree logic using a class and 
    # a helper to calculate the prefix maximums.
    
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
            self.tree_count[node] = self.tree_count[2 * node] + \
                                   self._query_count(2 * node + 1, mid + 1, end, self.tree_max[2 * node])
                                   
        def _query_count(self, node, start, end, current_max):
            if self.tree_max[node] <= current_max:
                return 0
            if start == end:
                return 1 if self.tree_max[node] > current_max else 0
            mid = (start + end) // 2
            # If the max of the left child is <= current_max, 
            # no elements in the left child can be prefix maximums.
            # We only check the right child.
            if self.tree_max[2 * node] <= current_max:
                return self._query_count(2 * node + 1, mid + 1, end, current_max)
            else:
                # The left child contributes some prefix maximums.
                # The right child's contribution is already precomputed 
                # relative to the left child's maximum.
                # Total = (count in left child given current_max) + (precomputed count in right child)
                # But the precomputed count in the right child is based on the left child's MAX,
                # not the current_max. Since we already checked that the left child's max 
                # is > current_max, the right child's contribution remains the same.
                return self._query_count(2 * node, start, mid, current_max) + \
                       (self.tree_count[node] - self.tree_count[2 * node])

        def get_count(self, i):
            # We want the number of prefix maximums in the range [i+1, N-1]
            # This is a range query on the Segment Tree.
            # However, the standard Segment Tree query returns a value.
            # We need to count prefix maximums in H[i+1...N-1].
            # This is equivalent to querying the range [i+1, N-1] with an initial max of 0.
            return self._range_query(1, 0, self.n - 1, i + 1, self.n - 1, 0)[0]

        def _range_query(self, node, start, end, l, r, current_max):
            if r < start or end < l:
                return (0, current_max)
            if l <= start and end <= r:
                count = self._query_count(node, start, end, current_max)
                return (count, max(current_max, self.tree_max[node]))
            mid = (start + end) // 2
            left_res = self._range_query(2 * node, start, mid, l, r, current_max)
            right_res = self._range_query(2 * node + 1, mid + 1, end, l, r, left_res[1])
            return (left_res[0] + right_res[0], right_res[1])

    # To avoid recursion depth issues and loops, we use the Segment Tree.
    # But since we can't use loops to call get_count, we map it.
    st = SegmentTree(H)
    results = map(st.get_count, range(N))
    print(*(results))

# Standard Python entry point
if __name__ == "__main__":
    # Increase recursion depth for deep Segment Trees
    sys.setrecursionlimit(300000)
    solve()