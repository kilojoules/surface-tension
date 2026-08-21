```python
import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # A building j is visible from i if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a new maximum 
    # encountered while scanning from i+1 to N, but the condition is 
    # slightly different: it's about the height of the target building j.
    
    # Let's re-evaluate: j satisfies the condition if max(H[i+1]...H[j-1]) < H[j].
    # This is exactly the definition of buildings that would be kept in a 
    # monotonic stack if we processed the array from right to left.
    # However, the standard "visible" problem is usually H[k] < H[i].
    # Here, the constraint is on H[j].
    
    # For a fixed i, we are looking for j > i such that H[j] > max(H[i+1]...H[j-1]).
    # This means j is a record-breaking height starting from index i+1.
    # The number of such j is the number of elements in the "upper envelope" 
    # of the sequence H[i+1...N].
    
    # We can solve this by processing the array from right to left.
    # We maintain a structure that allows us to count elements that are 
    # "prefix maximums" for any given starting point.
    # Since we need this for all i, and N is 2*10^5, an O(N log N) or O(N) is needed.
    
    # Let's use a Segment Tree or a similar structure. 
    # For a range [L, R], the number of visible buildings from L is:
    # 1 (for building L+1) + count_visible(L+1, R, height=H[L+1])
    
    # Using a recursive-like structure with a Segment Tree:
    # tree[node] stores the maximum height in that range.
    # query(node, current_max) returns the number of buildings in that range
    # that are taller than current_max and are record-breakers.
    
    # Since we cannot use loops, we use a functional approach to build the tree
    # and a helper to query it.
    
    def build_tree(l, r):
        if l == r:
            return (H[l], 1) # (max_val, count_of_visible_from_left)
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        
        # The number of visible buildings in the right child depends on the 
        # maximum of the left child.
        return (max(left[0], right[0]), 0) # We can't easily compute count here

    # The logic above is for a different problem. Let's use the property:
    # j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means for a fixed i, we are counting indices j > i that are 
    # left-to-right maximums of the suffix H[i+1...N].
    
    # This is a known problem solvable with a Segment Tree where each node 
    # stores the max of its range and a precomputed value: 
    # "how many elements in the right child are visible given the max of the left child".
    
    # Because of the "no loops" and "recursion depth" constraints, 
    # we implement the Segment Tree using a list and a functional query.
    
    # Given the constraints and the specific condition, we can use the fact 
    # that we need to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting elements in the suffix that are 
    # larger than all elements to their left (within that suffix).
    
    # Let's use a simpler observation: 
    # The answer for i is the number of elements in the set 
    # {j | i < j <= N and H[j] > max(H[i+1...j-1])}.
    # This is exactly the number of elements that would remain in a 
    # monotonic stack if we processed H[i+1...N] from left to right.
    
    # To do this for all i efficiently:
    # We can use a Segment Tree where each node stores:
    # 1. max_val: maximum height in the range.
    # 2. count(h): number of elements in the range that are visible 
    #    given a preceding maximum height h.
    
    # Since we can't define a function inside and call it recursively 
    # without hitting limits, we use a trick with a list-based Segment Tree.
    
    # However, there is a simpler way. The condition is:
    # j is counted for i if H[j] is a prefix maximum of the array H[i+1...N].
    # This is a classic problem. The answer for i is:
    # 1 + query(right_child, H[i+1])
    
    # To implement this without loops or deep recursion:
    # We use a Segment Tree stored in a list.
    # We use a helper function for the "count visible" logic.
    
    def get_visible(node_idx, l, r, current_max, tree_max, tree_count):
        # This is the core logic that would normally be recursive.
        # To avoid recursion, we can't easily implement this.
        # But wait, the constraints allow N=2*10^5. 
        # Maybe there's a different approach.
        pass

    # Let's reconsider: j is counted for i if H[j] > max(H[i+1...j-1]).
    # This means H[j] is a "left-to-right maximum" of the suffix starting at i+1.
    # This is equivalent to: j is counted for i if for all k such that i < k < j, H[k] < H[j].
    
    # Let's use the property: j is counted for i if and only if 
    # the nearest index k > j such that H[k] > H[j] is... no.
    # Actually, j is counted for i if the nearest index k < j such that H[k] > H[j] 
    # is less than or equal to i.
    # Let L[j] be the largest index k < j such that H[k] > H[j]. 
    # If no such k exists, L[j] = 0.
    # The condition "no building taller than Building j between i and j" 
    # is equivalent to: L[j] <= i.
    # Also we need i < j.
    # So for a fixed i, we need to count j such that: j > i AND L[j] <= i.
    
    # This is a 2D range counting problem: count j such that j > i and L[j] <= i.
    # We can solve this by iterating i from 1 to N.
    # As i increases, the condition L[j] <= i becomes easier to satisfy,
    # and the condition j > i becomes harder.
    
    # L[j] can be found using a monotonic stack in O(N).
    # Since we can't use loops, we use a functional approach to find L.
    # But we can use a simple list comprehension with a trick or 
    # just use the fact that we can use 'reduce' to simulate a stack.
    
    def find_L(heights):
        # Use reduce to simulate a monotonic stack to find the nearest larger to the left
        # stack stores indices.
        def step(state, current_idx):
            stack, result = state
            # Remove elements from stack that are smaller than current height
            # We can't use a while loop, so we use a helper function with recursion
            # But recursion depth is an issue. 
            # Actually, we can use a list comprehension to filter the stack? No.
            # Let's use a different approach for L[j].
            pass
        pass

    # Wait, the constraint H_i != H_j and 1 <= H_i <= N means H is a permutation.
    # Let's use the L[j] <= i and j > i logic.
    # Total count = (number of j such that L[j] <= i) - (number of j such that j <= i and L[j] <= i).
    # The second term is simply i, because for all j <= i, L[j] is always < j <= i.
    # So c_i = (number of j in [1, N] such that L[j] <= i) - i.
    
    # To find L[j] without loops:
    # We can use the 'divide and conquer' approach to find the nearest larger element.
    # Or, since we can use 'reduce', we can maintain the stack in the state.
    # To avoid the 'while' loop in the stack, we can use a recursive function 
    # for the popping part, as the total number of pops across the whole 
    # process is N, but a single step might pop many. 
    # However, the recursion depth for popping is only an issue if one 
    # element pops O(N) elements. We can use `sys.setrecursionlimit`.
    
    sys.setrecursionlimit(300000)
    
    def get_L(H):
        def pop_smaller(stack, h):
            if not stack or H[stack[-1]] > h:
                return stack
            return pop_smaller(stack[:-1], h)
        
        def reduce_func(state, idx):
            stack, L = state
            new_stack = pop_smaller(stack, H[idx])
            nearest_larger = new_stack[-1] + 1 if new_stack else 0
            # We use 1-based indexing for L as per the logic L[j] <= i
            # But the problem says Building 1...N. Let's use 0-based internally.
            # H is 0-indexed. Building j is index j-1.
            # L[j-1] is the index of the nearest building to the left taller than H[j-1].
            return (new_stack + [idx], L + [nearest_larger])
        
        # We need L[j] to be the 1-based index of the building.
        # If H[k] is the nearest taller, L[j] = k + 1.
        # The reduce_func above does this.
        C = reduce(reduce_func, range(N), ([], []))
        return C[1]

    L = get_L(H)
    
    # Now we need to count j such that L[j] <= i for each i in 1...N.
    # Let freq[v] be the number of j such that L[j] = v.
    # The number of j such that L[j] <= i is the prefix sum of freq.
    
    # Use a list to count frequencies of L values.
    # L values are in range [0, N-1].
    freq = [0] * (N + 1)
    # We can't use a loop to fill freq. We can use a technique with 
    # a dictionary and then mapping, or just use a list and 
    # a clever way to sum.
    # Actually, we can use a list comprehension to create a list of 
    # (L[j], 1) pairs and then sort them, but that's O(N log N).
    # A better way: use a list and then use a trick to sum.
    
    # Since we can't use loops, we can use a dictionary to count 
    # and then a list comprehension to build the freq array.
    from collections import Counter
    counts = Counter(L)
    freq = [counts.get(i, 0) for i in range(N + 1)]
    
    # Now compute prefix sums of freq to get the number of j where L[j] <= i.
    # We can use itertools.accumulate.
    from itertools import accumulate
    prefix_sum = list(accumulate(freq))
    
    # c_i = (number of j such that L[j] <= i) - i
    # Note: prefix_sum[i] is sum(freq[0...i]), which is count of L[j] <= i.
    # For i = 