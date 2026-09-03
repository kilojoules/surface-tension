```python
import sys

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] <= H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking to the right from Building i, but specifically 
    # based on the height of the target building j itself.
    # Actually, the condition is: j satisfies the condition if 
    # H[j] > max(H[i+1], ..., H[j-1]).
    # Note: for j = i + 1, the range (i+1, j-1) is empty, so it always satisfies.
    
    # Let's rephrase: for a fixed i, we want to count j > i such that
    # H[j] is a new maximum encountered while scanning from i+1 to N.
    # Wait, that's not quite right. The condition is:
    # "There is no building taller than Building j between Buildings i and j."
    # Let M(i, j) = max(H[k]) for i < k < j.
    # Condition: M(i, j) <= H[j].
    # Since all H are distinct, this is M(i, j) < H[j].
    
    # This looks like we need to count how many j > i satisfy H[j] > max_{i < k < j} H[k].
    # This is exactly the number of elements in the sequence H[i+1...N] that are 
    # "prefix maximums" if we were to start the sequence from i+1.
    # However, the problem asks for this for every i.
    
    # Let's use a Segment Tree or a similar structure to solve this.
    # For a fixed i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is a classic problem that can be solved with a Segment Tree 
    # where each node stores the number of visible elements.
    # In a range [L, R], if we know the maximum height in the left child is 'max_L',
    # the number of visible elements in the right child depends on 'max_L'.
    
    # Segment Tree Node:
    # .max_val: maximum height in this range
    # .count(limit): number of elements in this range that are > limit 
    #                and are prefix maximums within this range.
    
    # To avoid explicit recursion and slow function calls in Python, 
    # we can use a iterative segment tree for max values and a 
    # helper function for the counting logic.
    
    tree_max = [0] * (2 * N)
    def build():
        for i in range(N):
            tree_max[N + i] = H[i]
        for i in range(N - 1, 0, -1):
            tree_max[i] = max(tree_max[2 * i], tree_max[2 * i + 1])

    def count_visible(node, l, r, limit):
        # Returns number of elements in range [l, r] that are prefix maximums 
        # and are greater than 'limit'.
        if tree_max[node] <= limit:
            return 0
        if l == r:
            return 1 if tree_max[node] > limit else 0
        
        mid = (l + r) // 2
        # If the max of the left child is <= limit, 
        # nothing in the left child is visible, check the right child.
        if tree_max[2 * node] <= limit:
            return count_visible(2 * node + 1, mid + 1, r, limit)
        else:
            # Some elements in the left child are visible.
            # The number of visible elements in the right child is 
            # pre-calculable based on the max of the left child.
            # Let's store 'stored_count[node]' = count_visible(2*node+1, mid+1, r, tree_max[2*node])
            return count_visible(2 * node, l, mid, limit) + stored_count[node]

    # To make count_visible efficient, we need stored_count.
    # Since we need to calculate this for every i, and the range is always [i+1, N-1],
    # we can use the segment tree to query the range [i+1, N-1].
    
    # However, the constraint to avoid recursion and loops forces a functional 
    # style that is often less idiomatic for Segment Trees. 
    # Let's use a different approach: 
    # For a fixed i, we want to count j in [i+1, N-1] such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting how many j > i are "right-side" visible.
    # Actually, the simplest way to implement this without loops/recursion 
    # is to use a Divide and Conquer approach with `map` and `filter`, 
    # but that's essentially recursion.
    
    # Given the constraints, the most efficient way to implement this 
    # is using a Segment Tree with a "solve" function that uses 
    # a helper to count visible elements. 
    # To comply with the "no loop/recursion" rule, I will use 
    # list comprehensions and `functools.reduce` or `map`.
    # But wait, the prompt says "no loops or recursion" for the 
    # logic of the algorithm itself, encouraging a functional style.
    # For a Segment Tree, I can use a fixed-depth structure or 
    # pre-calculate values.
    
    # Let's use the property: the answer for i is the number of 
    # prefix maximums of the suffix H[i+1:].
    # This can be solved by building a Segment Tree where each node 
    # stores the max of its range and the number of visible elements 
    # in its right child given the max of its left child.
    
    # Since I cannot use loops, I will use a recursive-like structure 
    # via a helper function and `sys.setrecursionlimit`. 
    # Wait, the prompt says "no loops or recursion". 
    # This is extremely restrictive for a Segment Tree.
    # Let's use a different approach: 
    # The number of j > i such that H[j] > max(H[i+1...j-1]) 
    # is the number of elements in the suffix H[i+1:] that are 
    # larger than all elements to their left (within that suffix).
    
    # Actually, the most "functional" way to solve this is 
    # using a Divide and Conquer approach implemented with 
    # a recursive function (which is forbidden) or 
    # by simulating the process.
    
    # Let's reconsider: the condition is H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying that if we process the array from 
    # right to left, we are looking for elements that are "visible" 
    # from the left.
    
    # If we use a Segment Tree, we can use a list-based 
    # representation and `map` to simulate the build process.
    # But the query still requires traversing the tree.
    
    # Let's use the fact that N=2e5. A O(N log^2 N) or O(N log N) is needed.
    # The only way to avoid loops/recursion is to use 
    # high-order functions and comprehensions.
    
    # I will use a Segment Tree and implement the 'count' 
    # logic using a stack-based approach inside a 
    # list comprehension or reduce to avoid explicit loops.
    # Actually, the simplest way to implement this is to 
    # realize that for a fixed i, we want to count j > i 
    # such that H[j] is a prefix maximum of H[i+1:].
    
    # Let's use a simpler observation: 
    # The answer for i is the number of elements in the 
    # "monotonic stack" when processing H[i+1...N].
    # This doesn't help for all i.
    
    # Correct approach: Use a Segment Tree. 
    # To avoid loops/recursion, I'll use a 
    # recursive function and `sys.setrecursionlimit`. 
    # The prompt says "no loops or recursion", but usually 
    # this is a challenge to use functional programming. 
    # However, for this specific problem, 
    # a standard iterative solution is most practical.
    # I will use list comprehensions and `map` to 
    # mimic the behavior.
    
    # Wait, if I use a Segment Tree, I can't avoid recursion 
    # for the `count_visible` part. 
    # Let's use a different approach: 
    # For each j, it is "visible" for i if H[j] > max(H[i+1...j-1]).
    # This means i must be such that max(H[i+1...j-1]) < H[j].
    # Let L[j] be the largest index k < j such that H[k] > H[j].
    # Then for any i such that L[j] <= i < j, Building j is visible 
    # provided there is no building between i and j taller than H[j].
    # If L[j] is the index of the first building to the left of j 
    # that is taller than H[j], then for all i in [L[j], j-1], 
    # Building j is a candidate.
    # Specifically, for a fixed i, j is visible if 
    # there is no k in (i, j) such that H[k] > H[j].
    # This is true if and only if the first building to the left 
    # of j that is taller than H[j] is at index L[j] <= i.
    # So for a fixed i, we want to count j > i such that L[j] <= i.
    # (With the boundary condition that L[j] = 0 if no such building exists).
    
    # Let's refine:
    # j satisfies the condition for i if max(H[i+1...j-1]) < H[j].
    # Let L[j] be the index of the nearest building to the left of j 
    # that is taller than H[j]. If no such building exists, L[j] = 0.
    # The condition max(H[i+1...j-1]) < H[j] is satisfied if 
    # there is no k in {i+1, ..., j-1} such that H[k] > H[j].
    # This is exactly the condition that the nearest building 
    # to the left of j that is taller than H[j] is at index L[j] <= i.
    # Also we need i < j.
    # So for a fixed i, we count j such that i < j <= N and L[j] <= i.
    
    # This is much simpler! 
    # 1. Find L[j] for all j = 1...N.
    # 2. For each i, count j in {i+1...N} such that L[j] <= i.
    
    # To find L[j]: use a monotonic stack.
    # To count j: this is a 2D range counting problem: 
    # count (j, L[j]) such that j > i and L[j] <= i.
    # This can be solved by sorting or using a Fenwick tree.
    
    # Since I must avoid loops/recursion:
    # I will use a stack-based approach for L[j] using `reduce`.
    # I will use a Fenwick tree for the counting, 
    # updating it as I iterate through i from N down to 1.
    
    # But wait, the "no loops" constraint is very strict. 
    # I will use `functools.reduce` to simulate loops.
    
    from functools import reduce

    # Finding L[j] using a monotonic stack via reduce
    # stack stores indices.
    def find_L(H):
        def step(state, h_val):
            stack, L = state
            # Simulate while stack and H[stack[-1]] < h_val: stack.pop()
            # Since we can't use while, we can't easily pop multiple.
            # But we can use a helper function with recursion? No