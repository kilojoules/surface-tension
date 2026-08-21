import sys
from functools import reduce

def solve():
    # Read N and the heights
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
    # A building j is visible from i if it is a left-to-right maximum 
    # in the subarray H[i+1...N-1].
    
    # However, the problem asks for this for every i.
    # Let's rephrase: j is counted for i if max(H[i+1...j-1]) < H[j].
    # This is exactly the definition of elements that would remain in a 
    # monotonic stack if we processed the array from right to left.
    # Wait, that's not quite right. Let's use the property:
    # For a fixed i, we want the size of the set {j > i | H[j] > max(H[i+1...j-1])}.
    # This is simply the number of elements in the sequence H[i+1...N] 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # This looks like a range query problem, but we can solve it by observing
    # that we are counting indices j > i such that H[j] is a prefix maximum of 
    # the suffix starting at i+1.
    
    # Using a Segment Tree approach to count prefix maximums in a range:
    # For a range [L, R], the number of prefix maximums is:
    # 1 (the first element) + count_greater(L+1, R, H[L])
    # where count_greater(L, R, V) is the number of prefix maximums in [L, R]
    # that are also greater than V.
    
    tree_max = [0] * (4 * N)
    
    def build(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])

    def count_greater(node, start, end, V):
        if tree_max[node] <= V:
            return 0
        if start == end:
            return 1
        mid = (start + end) // 2
        # If the max of the left child is <= V, all prefix maximums 
        # must come from the right child.
        if tree_max[2 * node] <= V:
            return count_greater(2 * node + 1, mid + 1, end, V)
        # If the max of the left child > V, the right child's contribution
        # depends on the maximum of the left child.
        # The number of prefix maximums in the right child that are > max(left_child)
        # is precalculable: (total prefix maxes in node) - (prefix maxes in left child).
        return count_greater(2 * node, start, mid, V) + (tree_contrib[node] - tree_contrib[2 * node])

    # tree_contrib[node] stores the number of prefix maximums in the range of the node
    # relative to the start of its own range.
    tree_contrib = [0] * (4 * N)

    def build_with_contrib(node, start, end):
        if start == end:
            tree_max[node] = H[start]
            tree_contrib[node] = 1
            return
        mid = (start + end) // 2
        build_with_contrib(2 * node, start, mid)
        build_with_contrib(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        # The number of prefix maximums in the whole range is:
        # prefix maxes in left + prefix maxes in right that are > max(left)
        tree_contrib[node] = tree_contrib[2 * node] + count_greater(2 * node + 1, mid + 1, end, tree_max[2 * node])

    build_with_contrib(1, 0, N - 1)

    # For each i, we want the number of prefix maximums in the range [i+1, N-1].
    # We can use a helper function to query the number of prefix maximums in [L, R]
    # that are greater than a value V.
    def query(L, R, V):
        # This is a range query. Since we only need suffixes [i+1, N-1],
        # we can simplify. We need prefix maximums of H[L...N-1] that are > V.
        # We can implement this by traversing the segment tree.
        def recurse(node, start, end, L, R, V):
            if start > R or end < L or tree_max[node] <= V:
                return 0, V
            if start >= L and end <= R:
                # We need to count elements in this range > V.
                # But the V changes as we move across the range.
                # This specific logic is tricky. Let's use a simpler approach.
                pass
        pass

    # Actually, the simplest way to solve this is to realize that for a fixed i,
    # we are counting j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements that would be added to a 
    # monotonic stack when processing the suffix H[i+1...N-1] from left to right.
    # Since we need this for all i, and N=2e5, an O(N log^2 N) or O(N log N) is required.
    # The Segment Tree approach above (count_greater) is O(log^2 N) per query.
    # For a suffix [i+1, N-1], the answer is count_greater(1, 0, N-1, -1) 
    # restricted to the range [i+1, N-1].
    
    # Correct logic for suffix query:
    # The number of prefix maximums in [i+1, N-1] is simply 
    # count_greater(root, 0, N-1, H[i]) if we only consider indices > i.
    # Wait, the condition is: no building taller than H[j] between i and j.
    # This means H[k] < H[j] for all i < k < j.
    # This is exactly the definition of "visible" elements from index i.
    # The number of such j is the number of prefix maximums of the sequence H[i+1...N].
    
    # To find the number of prefix maximums in H[i+1...N-1]:
    # We can use the segment tree to query the range [i+1, N-1].
    def get_prefix_max_count(node, start, end, L, R, V):
        if start > R or end < L or tree_max[node] <= V:
            return 0, V
        if start >= L and end <= R:
            # We need to count how many in this range are prefix maximums
            # given the current maximum V.
            # We use the count_greater logic.
            res = count_greater(node, start, end, V)
            return res, max(V, tree_max[node])
        
        mid = (start + end) // 2
        left_count, left_v = get_prefix_max_count(2 * node, start, mid, L, R, V)
        right_count, right_v = get_prefix_max_count(2 * node + 1, mid + 1, end, L, R, left_v)
        return left_count + right_count, right_v

    # For each i from 0 to N-1, we query the range [i+1, N-1] with V = -1.
    results = [get_prefix_max_count(1, 0, N-1, i + 1, N - 1, -1)[0] for i in range(N)]
    print(*(results))

solve()