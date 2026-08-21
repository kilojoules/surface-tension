The constraint to avoid explicit loops forces a shift toward functional programming paradigms, which can lead to more concise code but is often less idiomatic in Python, as Python is designed to be multi-paradigm. To solve this problem without loops or recursion, I will use `itertools.accumulate` to maintain a running maximum of building heights and list comprehensions combined with `bisect_right` to efficiently count valid buildings. Specifically, for a fixed $i$, we are looking for $j > i$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is equivalent to saying that $H_j$ is a new maximum encountered as we scan from $i+1$ to $N$. However, since we need to do this for all $i$, a more efficient approach is to realize that for a fixed $i$, the valid $j$'s are the indices of the prefix maximums of the sequence $H_{i+1}, \dots, H_N$. 

Wait, the condition "no building taller than Building $j$ between $i$ and $j$" means $H_k < H_j$ for all $i < k < j$. This is exactly the definition of a building being "visible" from building $i$ looking right, provided we only count buildings that are taller than all buildings between them and the observer. Actually, the condition is simpler: $j$ is valid if $H_j > \max(H_{i+1}, \dots, H_{j-1})$. This means $j$ is a record-breaking height in the sequence starting from $i+1$.

Since $N=2 \times 10^5$, an $O(N^2)$ approach is too slow. We need $O(N \log N)$ or $O(N)$. The condition is satisfied if $j$ is the index of a prefix maximum of the suffix starting at $i+1$. This is a classic problem solvable with a monotonic stack. To avoid loops, I will use a Divide and Conquer approach implemented via `map` and list comprehensions, or more simply, realize that the number of such $j$ for a given $i$ is the number of elements in the monotonic stack after processing the suffix. However, since I cannot use loops, I will use a Segment Tree-like approach or a Fenwick tree implemented via `functools.reduce` or `map`, but that is overly complex. 

The most efficient way to solve this is to realize that for a fixed $i$, we are counting $j > i$ such that $H_j > \max_{i < k < j} H_k$. This is equivalent to counting how many $j$ are "visible" from $i$. This can be solved by iterating backwards and using a data structure. But since I cannot use loops, I will use `functools.reduce` to simulate the monotonic stack and store the results in a list.

```python
import sys
from functools import reduce

def solve():
    # Read N and heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find for each i, the number of j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to counting elements in a monotonic stack 
    # when processing the array from right to left.
    # Let's use reduce to maintain a stack and a list of counts.
    
    # The state for reduce: (stack, results)
    # stack: elements that could be a maximum for some i to their left
    # results: the count of visible buildings for each i
    
    def step(state, height):
        stack, results = state
        # We need to remove elements from the stack that are smaller than the current height
        # because the current height 'blocks' them from being seen by any i to the left.
        # However, the condition is: no building taller than Building j between i and j.
        # This means Building j is visible if it is a prefix maximum of the range [i+1, N].
        # For a fixed i, the valid j's are the indices that form the 
        # upper envelope of the heights to the right.
        
        # Wait, the condition is: H_k < H_j for all i < k < j.
        # This means for a fixed i, j is valid if H_j is a new maximum 
        # encountered while scanning from i+1 to N.
        # Example: 2 1 4 3 5
        # i=1 (H=2): j=2(H=1), j=3(H=4), j=5(H=5). Count=3.
        # i=2 (H=1): j=3(H=4), j=5(H=5). Wait, j=3 is 4, j=4 is 3, j=5 is 5.
        # For i=2, j=3(4), j=4(3) is NOT (since 4 > 3), j=5(5) is.
        # Actually, for i=2: j=3(H=4) is valid. j=4(H=3) is NOT because H_3(4) > H_4(3).
        # j=5(H=5) is valid because max(H_3, H_4) = 4 < 5.
        # So for i=2, j=3 and j=5 are valid. Count=2.
        
        # Correct logic: For a fixed i, we count j > i such that 
        # H_j > max(H_{i+1}, ..., H_{j-1}).
        # This is simply the number of prefix maximums of the sequence H[i+1...N].
        # Let f(i) be the number of prefix maximums of H[i+1...N].
        # f(N) = 0
        # f(N-1) = 1 (only j=N)
        # f(i): the first element H_{i+1} is always a prefix maximum.
        # Then we count how many prefix maximums are in H[i+2...N] that are > H_{i+1}.
        pass

    # To implement this without loops, we can use a Segment Tree or 
    # a similar structure to count elements > X in a range.
    # But the simplest way is to use the property:
    # The number of prefix maximums of H[i+1...N] is 1 + count_prefix_max(H[i+2...N], H_{i+1})
    # This looks like a functional data structure problem.
    # Given the constraints and the "no loop" rule, I will use a 
    # recursive-like structure via a Segment Tree implemented with 
    # list comprehensions and map, but that's complex.
    
    # Alternative: Use a Fenwick tree or Segment Tree to query 
    # the number of elements in the "upper envelope".
    # Actually, the number of prefix maximums of H[i+1...N] is 
    # the number of j > i such that H_j > max(H_{i+1}...H_{j-1}).
    # This is exactly the number of elements in the monotonic stack 
    # if we process from N down to 1.
    
    # Let's use a trick: the number of prefix maximums of H[i+1...N] 
    # is the number of elements in the monotonic stack of the suffix.
    # Since I can't use loops, I'll use a recursive-like approach 
    # with a balanced tree or just use the fact that 
    # I can use 'reduce' to simulate the stack.
    
    def solve_monotonic(acc, h_val):
        stack, res = acc
        # Remove elements from stack smaller than h_val
        # Since we can't loop, we use a slice or filter.
        # But we need to remove from the BOTTOM of the stack (the ones added last).
        # The stack contains the prefix maximums of the suffix.
        # When moving from i+1 to i, the new sequence is H_{i+1}, H_{i+2}...
        # The new prefix maximums are H_{i+1} and all prefix maximums of 
        # H[i+2...N] that are greater than H_{i+1}.
        
        # This is still tricky without loops. Let's use the property:
        # The number of prefix maximums of H[i+1...N] is 
        # 1 + (number of prefix maximums of H[i+2...N] that are > H_{i+1}).
        # This can be solved by building a Segment Tree where each node 
        # stores the number of prefix maximums.
        pass

# Given the strict "no loop" constraint, the most reliable way to 
# implement a complex logic is using map/reduce/comprehensions.
# However, for this specific problem, a Segment Tree is needed for O(N log N).
# I will implement a Segment Tree using list comprehensions.

import sys

# Increase recursion depth for deep trees
sys.setrecursionlimit(300000)

def solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # Segment Tree to count prefix maximums
    # tree[node] = max value in range
    # count(node, threshold) = number of prefix maximums in range > threshold
    
    tree_max = [0] * (4 * n)
    
    def build(node, start, end):
        if start == end:
            tree_max[node] = h[start]
            return
        mid = (start + end) // 2
        build(2 * node, start, mid)
        build(2 * node + 1, mid + 1, end)
        tree_max[node] = max(tree_max[2 * node], tree_max[2 * node + 1])

    def query_count(node, start, end, threshold):
        if tree_max[node] <= threshold:
            return 0
        if start == end:
            return 1
        mid = (start + end) // 2
        # If max of left child <= threshold, only right child can have prefix maxes
        if tree_max[2 * node] <= threshold:
            return query_count(2 * node + 1, mid + 1, end, threshold)
        # Otherwise, left child contributes some, and right child contributes
        # the number of prefix maxes it has that are > max(left child)
        # We can precompute the contribution of the right child.
        return query_count(2 * node, start, mid, threshold) + precomputed_right[node]

    # To avoid the loop in query_count and the recursion, 
    # and to satisfy the "no loop" constraint, I will use 
    # a different approach: the answer for i is the number of 
    # elements in the monotonic stack of the suffix.
    # I will use a recursive function with @lru_cache to simulate 
    # the process, but the constraint says no recursion.
    # Wait, the constraint says "no for/while loops or recursion".
    # This forces a purely functional approach.
    
    # Let's use the property: the answer for i is 
    # 1 + (number of j > i+1 such that H_j > max(H_{i+1}, max_{i+1 < k < j} H_k))
    # This is exactly the number of elements in the monotonic stack.
    # I can use a Segment Tree and a list comprehension to 
    # calculate the answers. But I can't build the tree without loops.
    
    # Actually, I can use `map` and `reduce` to build the tree.
    # But the simplest way to solve this is to realize that 
    # the number of visible buildings is the number of 
    # elements in the monotonic stack. 
    # I can use a linked-list style approach with a dictionary 
    # and a jump-pointer (like a sparse table) to count 
    # how many elements are larger.
    
    # Let next_greater[i] be the index of the first j > i such that H_j > H_i.
    # The number of visible buildings from i is 1 + count(next_greater[i+1])
    # where count(i) is the number of visible buildings starting from i.
    # This is a functional relationship: dp[i] = 1 + dp[next_greater[i]]
    
    # To find next_greater without loops:
    # We can use a divide and conquer approach with map/filter.
    # Or just use the fact that we can use a stack with reduce.
    
    def get_next_greater(n, h):
        # Use reduce to simulate a stack and find next greater element
        # state: (stack, next_greater_list)
        def reducer