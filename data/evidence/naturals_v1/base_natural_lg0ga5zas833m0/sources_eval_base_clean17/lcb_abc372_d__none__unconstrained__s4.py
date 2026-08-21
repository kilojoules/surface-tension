import sys

def solve():
    # Read N from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The problem asks for each i, how many j (i < j <= N) satisfy:
    # there is no building taller than Building j between i and j.
    # This means for a fixed i, we are looking for the number of elements 
    # to the right that are "visible" if we only consider elements that 
    # are strictly greater than all elements encountered so far since i.
    # Wait, the condition is: "no building taller than Building j between i and j".
    # Let's re-read: "between i and j" means indices k such that i < k < j.
    # The condition is: for all k such that i < k < j, H_k <= H_j.
    # Since H_i are distinct, this is H_k < H_j.
    
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum in the range [i+1, j].
    # Actually, a simpler interpretation: 
    # For a fixed i, we are counting j > i such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # Note that for j = i+1, the set of buildings between i and j is empty, 
    # so the condition is vacuously true.
    
    # This problem can be solved efficiently using a monotonic stack.
    # We want to find for each i, the number of j > i such that H_j is a 
    # prefix maximum of the sequence H_{i+1}, H_{i+2}, ..., H_N.
    
    # Let's process from right to left. 
    # For a fixed i, we want to count j in {i+1, ..., N} such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements in the sequence (H_{i+1}, ..., H_N)
    # that are strictly greater than all preceding elements in that sequence.
    
    # Let's use a Segment Tree or a similar structure? No, there's a simpler way.
    # The condition "H_j > max(H_{i+1}, ..., H_{j-1})" means that j is a 
    # record-breaking height starting from index i+1.
    # This is a classic problem. For a fixed range, the number of records
    # can be found. But we need it for all i.
    
    # Let's use a Monotonic Stack approach.
    # We process the array from right to left.
    # When we are at index i, we want to count how many j > i are "visible".
    # Building j is visible from i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # Let's maintain a monotonic decreasing stack of heights from the right.
    # Actually, consider the buildings to the right of i.
    # The first building j = i+1 is always visible.
    # The next visible building must be taller than H_{i+1}.
    # The one after that must be taller than the previous visible building.
    
    # Let's use a Segment Tree to find the number of elements that would be 
    # records. Or, we can use the property that the number of records in 
    # [i+1, N] is the same as the number of elements in the monotonic 
    # increasing stack if we processed [i+1, N] from left to right.
    
    # A better way:
    # For each i, we want to count j > i such that H_j > max_{i < k < j} H_k.
    # Let's use a Segment Tree where each node stores the maximum height in its range.
    # We can define a function `count_records(node, current_max)`:
    # If the max height in the node's range is <= current_max, return 0.
    # If the node is a leaf, return 1.
    # If the max height of the left child <= current_max, 
    # return count_records(right_child, current_max).
    # If the max height of the left child > current_max, 
    # return count_records(left_child, current_max) + (total_records_in_right_child - records_covered_by_left_child_max).
    
    # This is a known technique for "Range Record Queries".
    # Let's implement it.
    
    size = 1
    while size < n:
        size *= 2
    
    tree_max = [0] * (2 * size)
    tree_count = [0] * (2 * size)
    
    def update(i, val):
        idx = i + size
        tree_max[idx] = val
        # count is 1 for leaf if height > 0
        tree_count[idx] = 1 if val > 0 else 0
        idx //= 2
        while idx >= 1:
            tree_max[idx] = max(tree_max[2 * idx], tree_max[2 * idx + 1])
            tree_count[idx] = tree_count[2 * idx] + query_count(2 * idx + 1, tree_max[2 * idx])
            idx //= 2

    def query_count(node, current_max):
        if node >= 2 * size:
            return 0
        if tree_max[node] <= current_max:
            return 0
        if node >= size:
            return 1
        
        if tree_max[2 * node] <= current_max:
            return query_count(2 * node + 1, current_max)
        else:
            # The left child's max is > current_max.
            # The number of records in the right child that are > max of left child 
            # is already precalculated as (tree_count[node] - tree_count[2*node]).
            return query_count(2 * node, current_max) + (tree_count[node] - tree_count[2 * node])

    # Build the tree from right to left
    # For i = N, N-1, ..., 1
    results = [0] * n
    for i in range(n - 1, -1, -1):
        # For building i, we need records in range [i+1, N-1]
        # The range [i+1, N-1] is already updated in the tree.
        # We query the number of records in the whole tree starting with max = 0.
        # But the tree currently contains elements from i+1 to N-1.
        # Wait, we should query first, then update.
        
        # The range we care about is [i+1, N-1].
        # The query_count(1, 0) will give the number of records in the current tree.
        # However, the tree contains elements from i+1 to N-1.
        # Since we are processing right to left, we update the tree with H_i 
        # AFTER calculating for i? No, the condition is for j > i.
        # So for i, we query the tree containing {H_{i+1}, ..., H_{N-1}}.
        
        # However, the tree is built on indices 0...N-1.
        # To query range [i+1, N-1], we can just set elements 0...i to 0.
        # But we are processing right to left, so we can just add H_i to the tree
        # at position i, and then for the next i-1, the tree has H_i...H_{N-1}.
        
        # Let's refine:
        # For i = N-1 down to 0:
        #   1. result[i] = query_count(1, 0) # this queries the range [i+1, N-1]
        #   2. update(i, H[i])
        
        # Actually, query_count(1, 0) will count records in the entire tree.
        # Since we only updated indices i+1...N-1, and others are 0, it works.
        # But we need to be careful: the record counting is from left to right.
        # The tree is structured by index. So query_count(1, 0) counts records 
        # in the range [0, size-1]. Since we only filled [i+1, N-1], 
        # it will count records starting from the first non-zero element.
        
        # One problem: the records are counted from left to right.
        # The elements are at indices i+1, i+2, ..., N-1.
        # Our Segment Tree is built on indices 0...N-1.
        # So query_count(1, 0) will correctly count records starting from index i+1.
        
        # Let's trace:
        # i = N-1: result[N-1] = query_count(1, 0) = 0. Update(N-1, H[N-1])
        # i = N-2: result[N-2] = query_count(1, 0) = records in {H[N-1]}. Update(N-2, H[N-2])
        # i = N-3: result[N-3] = query_count(1, 0) = records in {H[N-2], H[N-1]}. Update(N-3, H[N-3])
        # This is correct.
        
        results[i] = query_count(1, 0)
        update(i, h[i])
        
    print(*(results))

if __name__ == "__main__":
    # Increase recursion depth for deep segment tree
    sys.setrecursionlimit(300000)
    solve()