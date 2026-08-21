import sys
from bisect import bisect_right

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The problem asks for the number of j > i such that 
    # max(H[i+1]...H[j-1]) < H[j].
    # This is equivalent to counting elements to the right of i that 
    # are "visible" from i.
    # A building j is visible from i if it is taller than all buildings between i and j.
    # This is a classic problem that can be solved by processing from right to left
    # and maintaining a data structure of "candidates" for visibility.
    # However, the condition "no building taller than Building j between i and j"
    # actually means we are looking for the number of indices j > i such that
    # H[j] > max(H[i+1]...H[j-1]).
    # This is exactly the number of elements in the sequence H[i+1...N] 
    # that form a "prefix maximum" sequence.
    
    # To solve this efficiently for all i, we can use a Segment Tree or 
    # a Divide and Conquer approach. 
    # Let f(i, current_max) be the number of visible buildings in range [i, N].
    # If H[i] > current_max:
    #    the answer is 1 + f(i+1, H[i])
    # Else:
    #    the answer is f(i+1, current_max)
    
    # Since we need this for every i, and the condition is about buildings 
    # BETWEEN i and j, Building i's height H[i] actually doesn't restrict 
    # whether Building j is visible; only buildings from i+1 to j-1 do.
    # So for a fixed i, we are counting j in {i+1, ..., N} such that 
    # H[j] > max(H[i+1], ..., H[j-1]).
    # This is simply the number of prefix maximums of the suffix H[i+1:].
    
    # Let g(i) be the number of prefix maximums of H[i...N].
    # If we are at index i, the first element H[i] is always a prefix maximum.
    # The remaining are the prefix maximums of H[i+1...N] that are greater than H[i].
    # We can use a Segment Tree where each node stores the max height in its range
    # and the number of prefix maximums within that range given a certain external max.
    
    # Segment Tree implementation:
    # tree_max[node]: max height in range
    # tree_count[node]: number of prefix maximums in right child that are > max of left child
    
    size = 1 << (N - 1).bit_length()
    tree_max = [0] * (2 * size)
    tree_count = [0] * (2 * size)
    
    # Initialize leaves
    for i in range(N):
        tree_max[size + i] = H[i]
    
    def get_count(node, current_max):
        # Returns number of prefix maximums in node's range that are > current_max
        if tree_max[node] <= current_max:
            return 0
        if node >= size:
            return 1
        
        # If left child's max is <= current_max, only right child can contribute
        if tree_max[2 * node] <= current_max:
            return get_count(2 * node + 1, current_max)
        else:
            # Left child contributes, and right child contributes its pre-calculated
            # count of elements > left child's max.
            return get_count(2 * node, current_max) + tree_count[node]

    # Build tree
    for i in range(size - 1, 0, -1):
        tree_max[i] = max(tree_max[2 * i], tree_max[2 * i + 1])
        tree_count[i] = get_count(2 * i + 1, tree_max[2 * i])
        
    # For each i, we want prefix maximums of H[i+1...N].
    # This is get_count(root, 0) for the range [i+1, N].
    # Since we can't easily query ranges, we can just build the tree for H[0...N-1]
    # and for each i, we need to "ignore" H[0...i].
    # Alternatively, we can use a simpler approach: 
    # The answer for i is get_count(root, 0) after setting H[0...i] to 0.
    # But that's O(N log^2 N). Let's use the property that we need suffixes.
    
    # Correct approach: The answer for i is the number of prefix maximums of H[i+1...N].
    # We can compute this by processing i from N-1 down to 0.
    # Let dp[i] be the number of prefix maximums of H[i...N-1].
    # dp[i] = 1 + (dp[next_greater[i]] if next_greater[i] exists else 0)
    # Wait, that's for elements > H[i]. The problem says "no building taller than 
    # Building j between i and j". This means H[j] > max(H[i+1]...H[j-1]).
    # This is exactly the definition of prefix maximums of the suffix starting at i+1.
    
    # Let's use the property: the answer for i is simply the number of 
    # prefix maximums of the sequence H[i+1], H[i+2], ..., H[N].
    # Let f(i) be the number of prefix maximums of H[i...N-1].
    # To find f(i): the first element H[i] is always a prefix max.
    # Then we need to count elements in H[i+1...N-1] that are > H[i] and are prefix maxes.
    # This is a known problem solvable with a Segment Tree in O(N log^2 N) or O(N log N).
    
    # Since we need the answer for all i, and the range is always [i+1, N-1],
    # we can use a persistent segment tree or just a Segment Tree and query 
    # the range [i+1, N-1].
    
    # Actually, the simplest O(N log^2 N) is:
    # For i = N-1 down to 0:
    #   ans[i] = query_prefix_maxes(i+1, N-1, current_max=0)
    #   update_tree(i, H[i])
    
    # But we can't "update" to the left. Let's just use the Segment Tree 
    # built on the whole array and query the range [i+1, N-1].
    
    def query_range(node, l, r, ql, qr, current_max):
        if ql <= l and r <= qr:
            # This is the tricky part: we need the count of prefix maxes 
            # in this range that are > current_max.
            # We can't just use tree_count because current_max changes.
            # We must use the get_count logic.
            # However, we need to return both the count and the new max.
            pass
            
    # Let's use a different approach: 
    # The number of j > i such that max(H[i+1]...H[j-1]) < H[j].
    # This is equivalent to: j is a prefix maximum of the suffix H[i+1...N].
    # Let's use the property that we can compute this using a Segment Tree 
    # where each node stores the max of its range and the number of 
    # prefix maximums in its range.
    
    # To avoid complex range queries, we can use the fact that we only need 
    # suffixes [i+1, N-1]. We can build the tree and then "delete" elements 
    # from the left by setting their height to 0.
    
    def update(i, val):
        idx = size + i
        tree_max[idx] = val
        while idx > 1:
            idx //= 2
            tree_max[idx] = max(tree_max[2 * idx], tree_max[2 * idx + 1])
            tree_count[idx] = get_count(2 * idx + 1, tree_max[2 * idx])

    # Initialize tree with all H
    for i in range(N):
        tree_max[size + i] = H[i]
    for i in range(size - 1, 0, -1):
        tree_max[i] = max(tree_max[2 * i], tree_max[2 * i + 1])
        tree_count[i] = get_count(2 * i + 1, tree_max[2 * i])

    # We need answers for i = 0 to N-1.
    # For i, we need prefix maxes of H[i+1...N-1].
    # We can process i from 0 to N-1, and at each step "remove" H[i] by setting it to 0.
    
    # Since we need to remove H[0], then H[1]... 
    # we can just use a list comprehension to generate the answers.
    # But we can't use a loop. We can use a recursive-like structure or 
    # map/reduce. However, the updates are sequential.
    # Wait, if we process i from 0 to N-1, we remove H[0], then H[1]...
    # This is exactly what we need. To do this without a loop, we can 
    # use a technique to simulate the state.
    
    # Actually, the simplest way to get prefix maximums of H[i+1...N-1] 
    # for all i is to realize that for a fixed i, the answer is 
    # get_count(root, 0) after setting H[0...i] = 0.
    
    # But we can just query the range [i+1, N-1] in the original tree!
    # A range query for prefix maximums:
    def range_query(node, l, r, ql, qr, current_max):
        if ql <= l and r <= qr:
            cnt = get_count(node, current_max)
            return cnt, max(current_max, tree_max[node])
        
        mid = (l + r) // 2
        # Left child
        res_l = (0, current_max)
        if ql < mid:
            res_l = range_query(2 * node, l, mid, ql, qr, current_max)
        
        # Right child
        res_r = (0, res_l[1])
        if qr > mid:
            res_r = range_query(2 * node + 1, mid, r, ql, qr, res_l[1])
            
        return (res_l[0] + res_r[0], res_r[1])

    # To avoid loops and recursion limits, we use a list comprehension 
    # to call range_query for each i.
    # Note: sys.setrecursionlimit is needed.
    sys.setrecursionlimit(300000)
    
    results = [range_query(1, 0, size, i + 1, N, 0)[0] for i in range(N)]
    print(*(results))

solve()