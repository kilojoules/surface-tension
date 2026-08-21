The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing what to calculate rather than how to iterate. In Python, this encourages the use of high-level abstractions and built-in iterators which are often more concise and can be more efficient. To solve this problem without loops, I will use a recursive approach (via a helper function) to simulate the traversal of the buildings and a list to store the results, utilizing `map` and `join` for the final output formatting.

```python
import sys

# Increase recursion depth for deep building lines
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # results will store c_i for each building
    results = [0] * N
    
    def process(i, stack):
        """
        We process the buildings from right to left.
        The stack maintains buildings that could be 'visible' to buildings to the left.
        Specifically, it stores indices of buildings in increasing order of height
        from the perspective of the current building i.
        Actually, a simpler approach: 
        For a fixed i, j satisfies the condition if H[k] < H[j] for all i < k < j.
        This means j is a 'right-side' visible building.
        The buildings j that satisfy this are exactly the ones that form a 
        strictly increasing subsequence of heights starting from the first building 
        to the right of i.
        
        Wait, the condition is: no building between i and j is taller than Building j.
        This means H[k] < H[j] for all k such that i < k < j.
        This is equivalent to saying that Building j is a 'prefix maximum' of the 
        sequence H[i+1...N].
        """
        pass

    # To avoid loops and recursion limits for the logic, 
    # we can use a Divide and Conquer approach or a Segment Tree, 
    # but those usually require loops for implementation.
    # However, the problem can be solved by observing that for a fixed i,
    # we are counting j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements in the sequence H[i+1...N] 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # Since I cannot use loops, I will use a recursive function to traverse 
    # the array and a stack-based logic implemented via recursion.
    # But the simplest way to count prefix maximums for all i is to realize:
    # j is counted for i if H[j] is the maximum of H[i+1...j].
    
    # Let's use a different perspective: 
    # For a fixed j, for which i < j is the condition satisfied?
    # The condition is H[k] < H[j] for all i < k < j.
    # This means i must be greater than the index of the first building to the 
    # left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the condition is satisfied for all i such that L[j] <= i < j.
    # The number of such i is j - L[j].
    
    # To find L[j] for all j without loops, we can use a recursive function 
    # that simulates the monotonic stack.
    
    def get_left_bounds(idx, stack, bounds):
        if idx == N:
            return bounds
        
        # Remove elements from stack that are smaller than current height
        # Using a helper to filter the stack without loops
        def filter_stack(s):
            if s and H[s[-1]] < H[idx]:
                return filter_s(s[:-1])
            return s
        
        # Since we can't use while loops, we use a recursive filter
        def pop_smaller(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return pop_smaller(s[:-1])
        
        current_stack = pop_smaller(stack)
        # L[j] is the index of the top of the stack
        left_bound = current_stack[-1] + 1 if current_stack else 0
        bounds.append(left_bound)
        
        return get_left_bounds(idx + 1, current_stack + [idx], bounds)

    # The above recursive approach is still O(N^2) in worst case due to slice 
    # and doesn't handle the stack efficiently. 
    # Let's use the property: c_i = count j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to: for a fixed i, we count how many times the 
    # running maximum of H[i+1...N] changes.
    
    # Given the constraints and the "no loop" rule, the most idiomatic 
    # way to solve this is using a Segment Tree or Fenwick Tree 
    # implemented via recursion, but that is overkill.
    
    # Let's use the L[j] logic. We need to find the nearest element to the left 
    # that is larger. This is a classic monotonic stack problem.
    # To implement a monotonic stack without loops, we can use 
    # a recursive function and a list as a stack.
    
    def compute_L(idx, stack, L):
        if idx == N:
            return L
        
        # Use a helper to simulate the 'while' loop of the monotonic stack
        def shrink(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return shrink(s[:-1])
        
        s_new = shrink(stack)
        L.append(s_new[-1] if s_new else -1)
        return compute_L(idx + 1, s_new + [idx], L)

    # To avoid recursion depth and slicing, we can use a trick with 
    # a custom class or a closure to maintain state, but the prompt 
    # forbids loops. We can use `map` and `functools.reduce`.
    from functools import reduce

    def step(state, idx):
        stack, L = state
        # Simulate while stack and H[stack[-1]] < H[idx]: stack.pop()
        # We use a recursive function inside reduce to simulate the while loop
        def pop_elements(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return pop_elements(s[:-1])
        
        new_stack = pop_elements(stack)
        L.append(new_stack[-1] if new_stack else -1)
        return (new_stack + [idx], L)

    # Using reduce to iterate through indices
    initial_state = ([], [])
    final_state = reduce(step, range(N), initial_state)
    L_bounds = final_state[1]
    
    # Now we have L[j] for each j.
    # c_i = count of j > i such that L[j] <= i.
    # This is equivalent to counting j in [i+1, N-1] such that L[j] <= i.
    # We can solve this by:
    # For each j, it contributes to c_i for i in [L[j], j-1].
    # This is a range update. We can use a difference array.
    
    diff = [0] * (N + 1)
    # For each j, increment range [L[j], j-1]
    # Since we can't use loops, we use map/reduce to update diff.
    def update_diff(d, j_L_pair):
        j, L_j = j_L_pair
        # Range is [L_j, j-1]. 
        # Note: L_j is 0-indexed here.
        # If L_j is -1, the range is [0, j-1].
        start = max(0, L_j)
        # We need to handle the case where L_j is the index of the taller building.
        # The condition is: no building taller than H[j] between i and j.
        # If L[j] is the index of the first building to the left taller than H[j],
        # then any i from L[j] to j-1 satisfies the condition.
        # Wait, if i = L[j], the building at L[j] is taller than H[j], 
        # but the condition says "between i and j". 
        # Buildings between i and j are indices k: i < k < j.
        # So if i = L[j], the buildings between are L[j]+1 ... j-1.
        # All these are smaller than H[j] by definition of L[j].
        # So i can be L[j].
        # The range of i is L[j] <= i < j.
        # However, i must be >= 0.
        # So i is in [max(0, L_j), j-1].
        
        # But we need to be careful: the problem says i < j.
        # And L[j] is the index of the first building to the left taller than H[j].
        # If L[j] is the index, then for any i such that L[j] <= i < j,
        # there is no building k (i < k < j) such that H[k] > H[j].
        # Because any such k would have to be <= L[j].
        # Since k > i >= L[j], k cannot be <= L[j] unless i < L[j].
        # So the range is indeed i \in [L[j], j-1].
        # Special case: if L[j] == -1, then i \in [0, j-1].
        
        # To implement range update [start, end] on diff array:
        # diff[start] += 1, diff[end+1] -= 1
        # But we can't mutate in a way that's not allowed? 
        # Actually, mutating a list inside reduce is generally accepted 
        # as long as the 'for' keyword isn't used.
        
        # Correcting the range:
        # If L[j] is the index of the first building to the left > H[j]:
        # For i = L[j], buildings between are L[j]+1 ... j-1. All are < H[j]. OK.
        # For i = L[j]-1, building L[j] is between i and j, and H[L[j]] > H[j]. NOT OK.
        # So i starts at L[j].
        
        # Handle L_j = -1 (no building to the left is taller)
        s = L_j if L_j != -1 else 0
        # The range of i is s <= i < j.
        # But the problem says i starts from 1 to N.
        # In 0-indexing, i is 0 to N-1.
        # The condition is i < j.
        # So for a fixed j, i can be s, s+1, ..., j-1.
        # This is (j-1) - s + 1 = j - s buildings.
        # These are the indices i.
        
        # To avoid the diff array and a second pass (which requires a loop),
        # we can just use the fact that c_i = sum_{j=i+1}^{N-1} [L[j] <= i].
        # This is a 2D counting problem: count pairs (i, j) such that 
        # 0 <= i < j < N and L[j] <= i.
        # This is sum_{j=1}^{N-1} (j - max(0, L[j])).
        # Wait, that's the total sum. We need it for each i.
        # c_i = count j such that j > i and L[j] <= i.
        # This is equivalent to: for a fixed i, count j in [i+1, N-1] such that L[j] <= i.
        
        # Let's use the diff array and then a prefix sum.
        # Since we can't use loops, we use a recursive function for prefix sum.
        return d

    # Re-evaluating: the most direct way to get c_i is:
    # c_i = (number of j > i such that L[j] <= i)
    # Let's use the property: c_i = (number