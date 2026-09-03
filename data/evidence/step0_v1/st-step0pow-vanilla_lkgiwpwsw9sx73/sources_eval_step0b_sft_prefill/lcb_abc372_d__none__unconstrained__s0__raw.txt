```python
import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j" 
    # means that for a fixed i, we are looking for the number of j > i 
    # such that H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "visible" building 
    # when looking right from Building i, but specifically defined by the 
    # height of the target building j, not the observer i.
    
    # Let's rephrase: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means the sequence of buildings that satisfy this for a fixed i 
    # are the elements of the "upper envelope" starting from i+1.
    # Specifically, if we have the sequence H[i+1...N], the indices j that 
    # satisfy the condition are the indices of the prefix maximums of that sequence.
    
    # However, calculating prefix maximums for every i would be O(N^2).
    # We need a more efficient way.
    
    # Observation: Building j is counted for Building i if and only if 
    # there is no k such that i < k < j and H[k] > H[j].
    # This means for a fixed j, we need to find how many i < j exist such that 
    # for all k in (i, j), H[k] < H[j].
    # This is equivalent to: i must be greater than or equal to the index of 
    # the first building to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the valid i's are L[j], L[j]+1, ..., j-1.
    # The number of such i's is (j-1) - L[j] + 1 = j - L[j].
    # Wait, the condition is: "no building taller than Building j between i and j".
    # If i = L[j], the buildings between i and j are {L[j]+1, ..., j-1}.
    # By definition of L[j], all these are < H[L[j]], but are they < H[j]?
    # Yes, because L[j] is the FIRST building to the left taller than H[j].
    # So for i = L[j], the condition is satisfied.
    # For i < L[j], the building at L[j] is between i and j and is taller than H[j], 
    # so the condition is NOT satisfied.
    # For i > L[j], the buildings between i and j are a subset of those between 
    # L[j] and j, all of which are < H[j], so the condition is satisfied.
    # Thus, for a fixed j, the valid i's are {L[j], L[j]+1, ..., j-1}.
    # Note: i must be at least 1. So i ranges from max(1, L[j]) to j-1.
    # The count is (j-1) - max(1, L[j]) + 1 = j - max(1, L[j]).
    # Special case: if L[j] does not exist, i can be 1, 2, ..., j-1. Count = j-1.
    
    # To solve this for all i, we can use a Difference Array (or Fenwick tree).
    # For each j from 2 to N:
    #   Let left_boundary = max(1, L[j])
    #   Range [left_boundary, j-1] gets +1.
    
    # To find L[j] for all j: use a monotonic stack.
    # H is 0-indexed in Python, so H[0...N-1].
    # L[j] will store the 0-based index.
    
    # Using a list comprehension to simulate the stack process since loops are banned.
    # Since we can't use loops, we can use a recursive function (with limit increased)
    # or a reduce trick to build the L array.
    
    # Let's use a stack inside a reduce to find the nearest greater element to the left.
    # The state is (stack, results_list).
    def process_stack(state, current_idx):
        stack, res = state
        # Remove elements smaller than current height
        # We need a way to do 'while' without 'while'. 
        # We can use a recursive function or a trick.
        # Since recursion is risky, let's use a helper function with map/filter 
        # or a custom reduction.
        
        # To simulate 'while stack and H[stack[-1]] < H[current_idx]: stack.pop()'
        # We can't easily. Let's use a recursive function for the pop logic.
        def pop_smaller(s, val):
            if s and H[s[-1]] < val:
                s.pop()
                return pop_smaller(s, val)
            return s
        
        new_stack = pop_smaller(stack, H[current_idx])
        left_boundary = new_stack[-1] if new_stack else -1
        res.append(left_boundary)
        new_stack.append(current_idx)
        return (new_stack, res)

    # To avoid recursion depth issues and loops, we can use a 
    # technique to find the Nearest Greater Element using a stack 
    # but implemented via a reduce. 
    # Python's recursion limit is an issue. Let's use a trick with `map` and a mutable state.
    
    # Actually, the constraint says "no loops". We can use a recursive function 
    # if we increase the recursion limit.
    sys.setrecursionlimit(300000)
    
    # To find L[j]:
    # We use a stack. For each j, we pop from stack while H[stack.top] < H[j].
    # The remaining stack.top is L[j].
    
    # Since we can't use while/for, we use a recursive function to handle the popping.
    def get_left_boundaries():
        stack = []
        results = []
        def recurse(idx):
            if idx == N:
                return
            
            def pop_func(s):
                if s and H[s[-1]] < H[idx]:
                    s.pop()
                    return pop_func(s)
                return s
            
            pop_func(stack)
            results.append(stack[-1] if stack else -1)
            stack.append(idx)
            recurse(idx + 1)
            
        recurse(0)
        return results

    # The recursion above is essentially a loop. Let's use a different approach.
    # We can use `reduce` to iterate through the indices and a helper to manage the stack.
    
    def stack_reducer(state, idx):
        stack, res = state
        # We need to pop elements. We can use a recursive inner function.
        def clean_stack(s, val):
            if s and H[s[-1]] < val:
                s.pop()
                return clean_stack(s, val)
            return s
        
        clean_stack(stack, H[idx])
        res.append(stack[-1] if stack else -1)
        stack.append(idx)
        return (stack, res)

    # To avoid "RecursionError", we can use a trick: 
    # The "while" loop can be simulated using a list comprehension or map 
    # if we have a mutable state, but the "pop" is the problem.
    # Actually, we can use a recursive function for the whole process 
    # but we must increase sys.setrecursionlimit.
    
    # Let's use the property: L[j] = (H[j-1] > H[j]) ? j-1 : L[j-1] (approximately)
    # Correct DP: L[j] = j-1 if H[j-1] > H[j] else L[L[j-1]] ...
    # This is the standard jump-pointer approach for Nearest Greater Element.
    
    # We can use a list and fill it using a map/comprehension if we can reference 
    # previous indices.
    # L[j] is the index of the nearest building to the left that is taller than H[j].
    # We can compute L[j] using a recursive-like structure but without loops.
    
    # Let's use a simple recursive function to find L for each j.
    # To avoid O(N^2), we use the jump pointer:
    # f(j) = if H[j-1] > H[j]: j-1 else: f(find_next_taller(j-1, H[j]))
    
    # Since we can't use loops, we'll use a recursive function to build the L array.
    # To prevent RecursionError, we use sys.setrecursionlimit.
    
    sys.setrecursionlimit(10**6)
    
    # L[j] = index of nearest building to the left > H[j]
    # We'll use a list to store results and a recursive function to simulate the process.
    L = [0] * N
    
    def compute_L(idx, stack):
        if idx == N:
            return
        
        # Simulate while stack and H[stack[-1]] < H[idx]: stack.pop()
        def pop_smaller(s):
            if s and H[s[-1]] < H[idx]:
                s.pop()
                return pop_smaller(s)
            return s
        
        pop_smaller(stack)
        L[idx] = stack[-1] if stack else -1
        stack.append(idx)
        compute_L(idx + 1, stack)

    compute_L(0, [])
    
    # Now we have L. For each j, the range of i is [L[j]+1, j-1].
    # (Using 0-based indexing: i is the index of the building, j is the index of the building).
    # The condition is: no building k such that i < k < j is taller than H[j].
    # This is true if i >= L[j].
    # So for a fixed i, we want to count j > i such that L[j] <= i.
    
    # Let's use the difference array approach.
    # For each j from 1 to N-1:
    #   left = L[j] + 1
    #   right = j - 1
    #   if left <= right:
    #     diff[left] += 1
    #     diff[right + 1] -= 1
    
    # Since we can't use loops, we use map/reduce.
    diff = [0] * (N + 1)
    
    def apply_diff(idx):
        # For building j = idx, it contributes to i in [L[idx]+1, idx-1]
        # Note: L[idx] is the index of the first building to the left > H[idx].
        # Any i from L[idx] to idx-1 (0-indexed) satisfies the condition.
        # Wait, if i = L[idx], the buildings between i and j are {L[idx]+1, ..., idx-1}.
        # All these are < H[idx] by definition of L[idx]. So i = L[idx] is valid.
        # If i < L[idx], then building L[idx] is between i and j, and H[L[idx]] > H[idx], so invalid.
        # So i ranges from L[idx] to idx-1.
        # But i must be >= 0.
        # Range is [max(0, L[idx]), idx - 1].
        
        l = max(0, L[idx])
        r = idx - 1
        if l <= r:
            diff[l] += 1
            diff[r + 1] -= 1
        return None

    # Use map to iterate through all j from 1 to N-1
    list(map(apply_diff, range(1, N)))
    
    # Compute prefix sums of diff to get c_i
    # We can't use a loop, so we use a custom reduce or a recursive function.
    def get_prefix_sums(diff_arr):