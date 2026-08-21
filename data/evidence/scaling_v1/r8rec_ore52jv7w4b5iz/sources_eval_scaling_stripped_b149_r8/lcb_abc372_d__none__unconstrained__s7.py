```python
import sys
from bisect import bisect_right

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to saying that Building j is a "visible" building
    # when looking right from Building i.
    # Specifically, the sequence of heights H[j] that satisfy this are the
    # running maximums of the suffix starting at i+1.
    # However, the condition is simpler: j satisfies the condition if 
    # H[j] is greater than all H[k] for i < k < j.
    # This means for a fixed i, the valid j's are those where H[j] 
    # is a new maximum encountered while scanning from i+1 to N.
    
    # To solve this efficiently for all i, we can use a Segment Tree or 
    # a Divide and Conquer approach. 
    # Let f(i, current_max) be the number of visible buildings to the right of i.
    # The number of visible buildings in range [L, R] given a height limit 'limit'
    # can be computed by:
    # 1. If max(range) <= limit, result is 0.
    # 2. If range is single element, result is 1 if H[mid] > limit else 0.
    # 3. Otherwise, split into [L, mid] and [mid+1, R].
    #    The right side's visibility depends on the max of the left side.
    
    # Since we need to avoid recursion depth and loops, we use a Segment Tree 
    # structure stored in a list and a clever way to query.
    # But the constraint N=2e5 and the "no loops" rule makes standard 
    # Segment Tree queries hard. 
    # Alternatively, we can observe that for a fixed i, we want the number of 
    # elements in the sequence H[i+1...N] that are strictly greater than 
    # all preceding elements in that subsequence.
    
    # This is a classic problem solvable by a Segment Tree where each node 
    # stores the max of its range and the number of visible elements 
    # relative to its own left child.
    
    # Because I cannot use while/for loops, I will use a Divide and Conquer 
    # approach implemented via a recursive-like structure or map, 
    # but Python's recursion limit is an issue.
    # Actually, the most efficient way to implement this without loops 
    # is to use a Segment Tree built and queried via recursion (with sys.setrecursionlimit).
    
    sys.setrecursionlimit(300000)
    
    # Build Segment Tree: tree[node] = (max_val, count_visible_from_left)
    # count_visible_from_left is the number of elements in the right child 
    # that are visible given the max of the left child.
    
    def build(l, r):
        if l == r:
            return (H[l], 1)
        mid = (l + r) // 2
        left_res = build(l, mid)
        right_res = build(mid + 1, r)
        
        # Helper to count visible elements in a range given a height limit
        def count_visible(node_l, node_r, limit):
            if node_l == node_r:
                return 1 if H[node_l] > limit else 0
            
            # This is the core logic of the "Segment Tree Beats" / Range Query
            # We need to access the children. Since we can't use a class/loop,
            # we'll store the tree in a dictionary.
            pass

    # To avoid the complexity of a custom Segment Tree without loops,
    # we can use the property that we want to count j > i such that 
    # H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting elements in the upper envelope.
    
    # Let's use a different approach: for each i, the answer is the number of 
    # elements in the set {j | j > i and H[j] > max(H[i+1...j-1])}.
    # This is exactly the number of elements that would remain if we 
    # processed the suffix [i+1, N] and kept only the increasing maximums.
    
    # We can solve this using a Segment Tree where each node stores:
    # 1. The maximum value in its range.
    # 2. The number of elements that are visible from the left of this range.
    
    tree_max = [0] * (4 * N)
    tree_vis = [0] * (4 * N)
    
    def build_tree(node, l, r):
        if l == r:
            tree_max[node] = H[l]
            tree_vis[node] = 1
            return
        mid = (l + r) // 2
        build_tree(2 * node, l, mid)
        build_tree(2 * node + 1, mid + 1, r)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        tree_vis[node] = calculate_visible(2 * node + 1, mid + 1, r, tree_max[2 * node])
        
    def calculate_visible(node, l, r, limit):
        if tree_max[node] <= limit:
            return 0
        if l == r:
            return 1
        mid = (l + r) // 2
        if tree_max[2 * node] <= limit:
            return calculate_visible(2 * node + 1, mid + 1, r, limit)
        else:
            # Left child has some visible, and the right child's visibility 
            # is already precomputed relative to the left child's max.
            return calculate_visible(2 * node, l, mid, limit) + (tree_vis[node] - tree_vis[2 * node])
            # Wait, the logic above is for a different problem. 
            # The correct logic for "visible from left" is:
            # tree_vis[node] = calculate_visible(right_child, limit = tree_max[left_child])
            # Total visible = calculate_visible(left_child, limit) + calculate_visible(right_child, max(limit, tree_max[left_child]))

    # Corrected logic for Segment Tree:
    # Each node stores:
    # .max: max height in range
    # .vis: number of elements in this range visible from the left of THIS range.
    
    def get_vis(node, l, r, limit):
        if tree_max[node] <= limit:
            return 0
        if l == r:
            return 1
        mid = (l + r) // 2
        if tree_max[2 * node] <= limit:
            return get_vis(2 * node + 1, mid + 1, r, limit)
        else:
            # Left child contributes some, and right child's contribution 
            # is already stored as (tree_vis[node] - get_vis(left_child, limit=left_child_max))
            # Actually: tree_vis[node] is get_vis(right_child, tree_max[left_child])
            # So total is get_vis(left_child, limit) + tree_vis[node]
            return get_vis(2 * node, l, mid, limit) + tree_vis[node]

    def build_final(node, l, r):
        if l == r:
            tree_max[node] = H[l]
            tree_vis[node] = 0 # Not used for leaves in this logic
            return
        mid = (l + r) // 2
        build_final(2 * node, l, mid)
        build_final(2 * node + 1, mid + 1, r)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])
        tree_vis[node] = get_vis(2 * node + 1, mid + 1, r, tree_max[2 * node])

    build_final(1, 0, N - 1)
    
    # For each i, we want visible elements in range [i+1, N-1] with limit = -1
    # (since the condition is no building taller than Building j BETWEEN i and j,
    # Building i itself doesn't block Building i+1).
    
    def query(node, l, r, ql, qr, limit):
        if ql <= l and r <= qr:
            return get_vis(node, l, r, limit), tree_max[node]
        mid = (l + r) // 2
        # This is tricky without loops. We must handle the range query.
        # But we only need suffixes [i+1, N-1].
        # For a suffix, we can just query the range and update the limit.
        pass

    # Since we need suffixes, we can use a simpler approach.
    # The answer for i is get_vis(root, 0, N-1, limit=H[i]) 
    # but only for the range [i+1, N-1].
    # Actually, the condition "no building taller than Building j between i and j"
    # means Building j is a running maximum of the sequence H[i+1...N].
    # The height of Building i does NOT matter.
    
    # Let's redefine: for each i, we want the number of running maximums in H[i+1...N].
    # This is exactly what the Segment Tree 'get_vis' does for the range [i+1, N-1] with limit=0.
    
    def query_suffix(node, l, r, ql, limit):
        if l == r:
            return (1 if H[l] > limit else 0, max(limit, H[l]))
        mid = (l + r) // 2
        if ql <= l:
            # Entire left child is in range
            # We need to know how many in left are visible, then how many in right.
            # This is still recursive.
            pass

    # To avoid the range query complexity, we can use the fact that we only need suffixes.
    # We can use a Divide and Conquer function:
    # solve(l, r) returns a list of answers for i in [l, r].
    
    def solve_dc(l, r):
        if l == r:
            return [0]
        mid = (l + r) // 2
        # Divide
        left_ans = solve_dc(l, mid)
        right_ans = solve_dc(mid + 1, r)
        
        # For i in [l, mid], the buildings in [mid+1, r] that are visible
        # are those taller than max(H[i+1...mid]) and taller than 
        # any building between mid and j.
        # This is still complex. Let's use the Segment Tree get_vis on the whole range
        # and just handle the suffix by querying the range [i+1, N-1].
        
        # Wait, the simplest way to implement this is to use the property:
        # ans[i] = count_visible(range=[i+1, N-1], limit=0)
        # We can implement a function that queries the suffix [i+1, N-1].
        
        return []

    # Final attempt: Use the Segment Tree and a helper to query suffixes.
    # Since I can't use loops, I'll use a list comprehension to generate answers.
    # To make query_suffix work, I'll use the property that I can query the 
    # whole range [0, N-1] and the Segment Tree will naturally ignore 
    # indices < i if I pass a very large limit for that range.
    # But that's not possible. 
    
    # Let's use the property: ans[i] = get_vis(root, 0, N-1, limit=0) 
    # but only counting indices > i.
    # Actually, the most straightforward way is to use a recursive function 
    # with a decorator for caching or just pure recursion.
    
    def get_ans(i):
        # Count running maximums in H[i+1 : N]
        # We can use a helper function that processes the Segment Tree.
