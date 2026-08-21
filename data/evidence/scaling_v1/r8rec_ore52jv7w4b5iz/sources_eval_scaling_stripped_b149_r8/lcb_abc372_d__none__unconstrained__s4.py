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
    
    # The problem asks for the number of j > i such that 
    # max(H[i+1]...H[j-1]) < H[j].
    # This is equivalent to counting elements to the right that are 
    # "visible" from index i.
    # An element H[j] is visible from i if it is larger than all elements between i and j.
    # This means we are looking for the number of elements in the sequence 
    # H[i+1...N-1] that form a strictly increasing subsequence of "running maximums".
    
    # However, the constraint is: no building taller than Building j between i and j.
    # Let's re-evaluate: for a fixed i, we count j > i such that for all k (i < k < j), H[k] < H[j].
    # This is exactly the definition of elements that would be added to a monotonic 
    # stack if we processed the array from i+1 to N.
    # But we need this for all i.
    
    # Let's use the property: H[j] satisfies the condition for i if 
    # H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying that if we look at the sequence from right to left,
    # we want to count how many H[j] are "reachable".
    
    # Actually, the condition "no building taller than Building j between i and j"
    # means H[j] > max({H[k] | i < k < j}).
    # If i = N, the answer is 0.
    # If i = N-1, j = N is the only possibility. H[N] > max(empty set) is always true. Answer: 1.
    # For a general i, the buildings j that satisfy this are those that 
    # would be part of the "upper hull" of the sequence starting from i+1.
    
    # Let's use a Segment Tree or a similar structure to count.
    # For a fixed i, we are looking for the number of j > i such that 
    # H[j] is a prefix maximum of the sequence H[i+1...N].
    # Wait, the condition is: max(H[i+1...j-1]) < H[j].
    # This means H[j] is a "new maximum" encountered while scanning from i+1 to N.
    # The number of such j is simply the number of times the prefix maximum changes
    # when scanning H from index i+1 to N.
    
    # This is a classic problem that can be solved with a Segment Tree.
    # Each node in the Segment Tree will store:
    # 1. The maximum value in its range.
    # 2. A value 'count' which is the number of prefix maximums in its range,
    #    given the maximum of the range to its left.
    
    # Since we cannot use loops, we use a recursive-like structure via a Segment Tree.
    # But Python's recursion limit and speed are issues. 
    # Let's implement the Segment Tree using a list and a helper function.
    
    # tree_max[v] stores max(H[l...r])
    # tree_cnt[v] stores the number of prefix maximums in the right child 
    # that are greater than the maximum of the left child.
    
    # To avoid recursion, we can build the tree iteratively.
    # However, the 'query' part (counting prefix maximums) is naturally recursive.
    # We can use a trick: the number of prefix maximums in range [l, r] 
    # given a threshold 'T' can be computed by:
    # if max(left_child) <= T: return solve(right_child, T)
    # if max(left_child) > T: return solve(left_child, T) + (tree_cnt[v])
    
    # Given the constraints and Python, a simpler approach might be needed.
    # Let's use the Segment Tree logic but implement the 'solve' part 
    # using a stack-based approach or a clever observation.
    
    # Actually, the number of j for a fixed i is simply the number of 
    # elements in the set {j | j > i and H[j] > max(H[i+1...j-1])}.
    # This is exactly the number of elements that would remain in a 
    # monotonic stack if we pushed H[i+1...N] onto it.
    
    # For N=2e5, O(N log N) or O(N log^2 N) is required.
    # The Segment Tree approach is O(N log N).
    
    # Since I must provide a complete working solution without loops:
    # I will use a Segment Tree implemented with list comprehensions and 
    # a recursive function for the prefix maximum count (increasing recursion depth).
    
    sys.setrecursionlimit(300000)
    
    def build(l, r):
        if l == r:
            return (H[l], 0) # (max, cnt)
        mid = (l + r) // 2
        left = build(l, mid)
        right = build(mid + 1, r)
        # cnt is the number of prefix maxes in the right subtree 
        # that are greater than the max of the left subtree.
        return (max(left[0], right[0]), count_greater(right, left[0], mid + 1, r))

    def count_greater(node, threshold, l, r):
        # This is the core of the Segment Tree approach to count prefix maximums.
        # Because we need to avoid loops and recursion is allowed (with limit),
        # we define this logic. But 'node' needs to be the actual tree structure.
        pass

    # To avoid the complexity of a custom Segment Tree in Python, 
    # let's use the property: we want to count j > i such that 
    # H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting j > i such that 
    # for all k from i+1 to j-1, H[k] < H[j].
    
    # Let's use a different approach: 
    # For each j, it is "visible" from i if H[j] > max(H[i+1...j-1]).
    # This means i+1 must be greater than the index of the first building 
    # to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0.
    # Building j satisfies the condition for i if i >= L[j] and i < j.
    # So for a fixed i, we need to count j such that j > i and L[j] <= i.
    
    # L[j] can be found using a monotonic stack in O(N).
    # Then we need to count j in range [i+1, N] such that L[j] <= i.
    # This is a 2D range counting problem: (j, L[j]) such that j > i and L[j] <= i.
    # Since we can't use loops, we can use a Fenwick tree and process queries offline.
    # But we can't use loops to iterate through the array.
    # We can use `map` or `reduce` from functools.
    
    from functools import reduce
    
    # 1. Find L[j] for all j using a stack-based approach with reduce.
    # stack stores indices of buildings.
    def find_L(acc, curr_idx):
        stack, L = acc
        # Remove elements from stack that are smaller than current height
        # Since we can't use while, we use a helper function with recursion
        def pop_smaller(s):
            if s and H[s[-1]] < H[curr_idx]:
                return pop_smaller(s[:-1])
            return s
        
        new_stack = pop_smaller(stack)
        nearest_left = new_stack[-1] + 1 if new_stack else 1
        return (new_stack + [curr_idx], L + [nearest_left])

    # To avoid recursion depth issues in pop_smaller, we can't use it.
    # Let's use a different way to find L[j].
    # L[j] is the index of the first element to the left > H[j].
    # We can use a Segment Tree to find the rightmost index k < j such that H[k] > H[j].
    
    # Actually, the simplest way to implement this in Python without loops 
    # is to use a Fenwick tree and process indices using map/reduce.
    # But we still need L[j].
    
    # Let's use the property: L[j] can be found by sorting buildings by height.
    # Or use a Divide and Conquer approach.
    
    # Final attempt: Use the L[j] logic. To find L[j] without loops:
    # We can use a recursive function with a helper to simulate the stack.
    # But the most reliable way is to use a Fenwick tree and a sorted list of events.
    
    # Since I must return only the code block and it must be complete:
    # I will use a Divide and Conquer approach to count pairs (i, j).
    
    def solve_dc(l, r):
        if l == r:
            return [0]
        mid = (l + r) // 2
        # This is getting complex. Let's use the L[j] logic with a trick.
        # We can find L[j] by using a Segment Tree (implemented via a list)
        # and performing a binary search on the tree.
        pass

    # Given the constraints and the "no loop" rule, the most Pythonic 
    # way to handle this is using a Fenwick tree and `reduce`.
    
    # To find L[j] without loops:
    # We can use the fact that H_i are distinct and in range [1, N].
    # We can use a Fenwick tree to find the maximum index k < j with H[k] > H[j].
    # But that's also O(N log N).
    
    # Let's use the most direct interpretation:
    # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to: count j > i such that L[j] <= i.
    # Where L[j] is the index of the nearest building to the left of j taller than H[j].
    
    # To find L[j] for all j in O(N log N) without loops:
    # We can use a Segment Tree to find the rightmost index in [1, j-1] with value > H[j].
    
    # Actually, the simplest O(N log N) is to use a Fenwick tree 
    # and process queries by sorting.
    
    # Let's implement the L[j] calculation using a recursive function 
    # and the counting using a Fenwick tree.
    
    def get_L(indices, stack):
        if not indices:
            return []
        curr = indices[0]
        # Simulate while stack and H[stack[-1]] < H[curr]: stack.pop()
        # Using a helper to avoid loops
        def clean_stack(s):
            if s and H[s[-1]] < H[curr]:
                return clean_stack(s[:-1])
            return s
        
        s_cleaned = clean_stack(stack)
        res = s_cleaned[-1] + 1 if s_cleaned else 1
        return [res] + get_L(indices[1:], s_cleaned + [curr])

    # The above is still recursive. Let's use a more robust approach.
    # We can find L[j] by processing heights in descending order 
    # and using a SortedList (from bisect) to find the nearest index to the left.
    
    # 1. Create pairs (H[i], i) and sort them descending.
    # 2. Use a SortedList to keep track of indices of buildings already processed.
    # 3. For the current building at index i, the nearest taller building to the left 
    #    is the element in the SortedList just smaller than i.
    
    from bisect import insort, bisect_left
    
    # We need to avoid loops. We can use `