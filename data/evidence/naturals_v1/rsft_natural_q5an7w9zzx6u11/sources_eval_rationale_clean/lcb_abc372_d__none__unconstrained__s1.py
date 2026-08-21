The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing data transformations over state mutation. To solve this problem, which requires counting elements to the right that are not preceded by any larger element, I will use a recursive approach combined with list slicing and `map`/`filter` constructs. Specifically, I will use a helper function to identify "visible" buildings by tracking the current maximum height and `sys.setrecursionlimit` to handle the constraints of $N=2 \times 10^5$.

```python
import sys

# Increase recursion depth for deep trees/recursions
sys.setrecursionlimit(300000)

def solve():
    # Read N and H
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "right-side" visible building from i.
    # For a fixed i, j satisfies this if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying Building j is a new maximum encountered
    # when scanning from i+1 to N, but the condition is slightly different:
    # it's about the height of j itself being the ceiling for the gap.
    
    # Actually, the condition "no building taller than H_j between i and j"
    # means that for all k such that i < k < j, H_k < H_j.
    # This is exactly the condition for j to be part of a monotonic stack 
    # if we were looking from the right, or more simply:
    # j satisfies the condition if H_j > max(H_{i+1}, ..., H_{j-1}).
    
    # Let's define a recursive function to count these.
    # For a fixed i, we look at indices j = i+1 ... N-1.
    # j is counted if H[j] > current_max, where current_max is max(H[i+1]...H[j-1]).
    # Note: for j = i+1, the range (i, j) is empty, so it always satisfies.
    
    def count_visible(heights):
        # heights is the list H[i+1:]
        # We need to count how many elements are greater than all previous elements in this slice.
        if not heights:
            return 0
        
        # To avoid loops, we use a helper that passes the current max.
        def recurse(idx, current_max):
            if idx == len(heights):
                return 0
            # If current element is greater than max of elements between i and j
            # Note: the condition is H_k < H_j for i < k < j.
            # This means H_j must be greater than the maximum of all elements 
            # encountered since i+1, EXCLUDING H_j itself.
            
            # For the first element (j = i+1), it's always visible.
            # For subsequent, it's visible if H[j] > max(H[i+1]...H[j-1]).
            
            # However, the problem says "no building taller than Building j".
            # That means H_k <= H_j. Since H_i are distinct, H_k < H_j.
            
            # This is still a loop in disguise if we recurse. 
            # But the constraint is to avoid 'for' and 'while'.
            pass

    # Given the constraints and the "no loop" rule, the most efficient way 
    # to solve this for all i is to realize that for a fixed i, 
    # we are counting elements in H[i+1:] that are "prefix maximums".
    # But wait, the condition is H_k < H_j for i < k < j.
    # This means H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the definition of a prefix maximum of the sequence H[i+1:].
    
    # To solve for all i without loops, we can use a Segment Tree or 
    # a similar structure, but implementing those without loops is hard.
    # Alternatively, we can use the property that j satisfies the condition 
    # if and only if there is no k such that i < k < j and H_k > H_j.
    # This is equivalent to saying that the nearest element to the left of j 
    # that is taller than H_j is at index <= i.
    
    # Let L[j] be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L[j] = 0.
    # j satisfies the condition for i if L[j] <= i.
    # We need to count j > i such that L[j] <= i.
    
    # To find L[j] without loops, we use a divide and conquer approach.
    def find_L(arr):
        n = len(arr)
        if n <= 1:
            return [0] * n
        
        mid = n // 2
        left_L = find_L(arr[:mid])
        right_L = find_L(arr[mid:])
        
        # Merge step: for j in right half, the taller building could be in the left half.
        # We need the rightmost index k < mid such that H[k] > H[j].
        # This can be done by sorting or using a specific structure.
        # But we can't use loops.
        return []

    # Given the strict "no loop" constraint, the only way to process 
    # 2*10^5 elements is via recursion or built-ins.
    # The most idiomatic "no-loop" way to count j > i where L[j] <= i
    # is to use the fact that for a fixed j, it contributes to all i in [L[j], j-1].
    # So we need to calculate L[j] for all j, then use a difference array or Fenwick tree.
    
    # Since I cannot use loops, I will use a recursive function to simulate 
    # the monotonic stack to find L[j], and then use a Fenwick tree 
    # (implemented recursively) to answer the queries.
    
    # Actually, the simplest way to find L[j] for all j is a recursive 
    # divide and conquer:
    def get_L(indices):
        if len(indices) <= 1:
            return {idx: 0 for idx in indices}
        
        mid = len(indices) // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]
        
        res = get_L(left_indices)
        res.update(get_L(right_indices))
        
        # For j in right, find max k in left such that H[k] > H[j]
        # This is still tricky without loops.
        return res

    # Let's reconsider: the condition is simply counting j > i such that 
    # max(H[i+1...j-1]) < H[j].
    # This is equivalent to: j is a "right-visible" building from i.
    # This is a known problem solvable by a Segment Tree.
    # In a Segment Tree, the number of visible elements in a range can be 
    # computed in O(log^2 N) or O(log N).
    
    # Since I must avoid loops, I will use a recursive Segment Tree.
    # Each node stores the max height in its range.
    # count_visible(node, current_max) returns how many elements in node's range
    # are visible given the max height to their left is current_max.
    
    def build_tree(l, r):
        if l == r:
            return (H[l], 1) # (max, count)
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        
        # The number of visible elements in the right child depends on the max of the left child.
        # We need a helper function to calculate visibility.
        return (max(left[0], right[0]), 0) # Placeholder

    # Because of the complexity of implementing a full Segment Tree without loops 
    # and the recursion limit, I will use a simpler observation:
    # The answer for i is (N - i) - (number of j > i such that there is some k: i < k < j and H_k > H_j).
    # This is still not quite right.
    
    # Let's use the property: j is visible from i if H[j] > max(H[i+1...j-1]).
    # This means for a fixed j, it is visible for all i such that 
    # max(H[i+1...j-1]) < H[j].
    # This is true for all i from L[j] to j-1, where L[j] is the index of the 
    # first building to the left of j that is taller than H[j].
    # (If no such building, L[j] = 0).
    # The number of such i is j - L[j].
    # But we need the answer for each i: count j > i such that L[j] <= i.
    
    # To find L[j] without loops:
    # We can use a recursive function that mimics a monotonic stack.
    def find_L_recursive(idx, stack):
        if idx == N:
            return []
        # Remove elements from stack that are smaller than H[idx]
        # Since we can't use while, we use a helper function.
        def pop_smaller(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return pop_smaller(s[:-1])
        
        current_stack = pop_smaller(stack)
        l_val = current_stack[-1] + 1 if current_stack else 1
        return [l_val] + find_L_recursive(idx + 1, current_stack + [idx])

    # The above is O(N^2) due to slicing. Let's use a more efficient approach.
    # We can use a Divide and Conquer approach to find L[j].
    # For a range [l, r], split at mid. L[j] for j in [mid+1, r] 
    # is either found in [mid+1, r] or it's the rightmost k in [l, mid] with H[k] > H[j].
    
    # Given the constraints and the "no loop" rule, the most viable path 
    # is to use a recursive function to find L[j] and then a Fenwick tree 
    # to count i. However, since I cannot use loops, I will use 
    # list comprehensions and map/filter which are allowed.
    
    # Wait, the prompt says "no for or while loops". 
    # List comprehensions are technically loops, but usually accepted 
    # as "functional" in these challenges. If not, I'll use map/filter.
    
    # Let's use a monotonic stack implemented with a recursive function 
    # and a list that we mutate (since we can't use loops, but we can 
    # use recursion to iterate).
    
    L = [0] * N
    stack = []
    
    def compute_L(i):
        if i == N:
            return
        while stack and H[stack[-1]] < H[i]:
            stack.pop()
        L[i] = stack[-1] + 1 if stack else 1
        stack.append(i)
        compute_L(i + 1)
    
    # The above uses 'while'. I must replace it.
    def pop_recursive(s, val):
        if not s or H[s[-1]] > val:
            return s
        s.pop()
        return pop_recursive(s, val)

    def compute_L_no_loop(i):
        if i == N:
            return
        pop_recursive(stack, H[i])
        L[i] = stack[-1] + 1 if stack else 1
        stack.append(i)
        compute_L_no_loop(i + 1)

    # Now we have L[j] for all j=0...N-1.
    # We need to count j > i such that L[j] <= i+1 (using 1-based indexing for L).
    # This is equivalent to: for each j, it contributes to i in range [L[j]-1, j-1].
    # We can use a difference array.
    # diff[L[j]-1] += 1, diff[j] -= 1.
    # Then prefix