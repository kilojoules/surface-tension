The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm, emphasizing data transformations over state mutation. To solve this problem without loops, I will use a recursive approach (simulating a stack-based traversal) combined with `map` and `list` comprehensions, utilizing Python's `sys.setrecursionlimit` to handle the input size.

```python
import sys

# Increase recursion depth for deep trees/stacks
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # Specifically, j satisfies the condition if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying Building j is part of the 
    # upper-envelope of the heights to the right of i.
    
    # However, a simpler observation:
    # For a fixed i, the indices j that satisfy the condition are those that
    # would be pushed onto a monotonic stack when iterating from i+1 to N.
    # But we need this for all i.
    
    # Let's redefine: j satisfies the condition for i if 
    # H[j] > max(H[i+1] ... H[j-1]).
    # This means for a fixed j, it contributes to all i < j such that
    # max(H[i+1] ... H[j-1]) < H[j].
    
    # Let L[j] be the nearest index k < j such that H[k] > H[j].
    # Then for all i such that L[j] <= i < j, Building j satisfies the condition.
    # (If no such k exists, L[j] = 0).
    # The number of such i is j - L[j].
    
    # We can find L[j] for all j using a monotonic stack.
    # Since we cannot use loops, we use recursion to simulate the stack.
    
    def find_L(idx, stack, results):
        if idx == N:
            return results
        
        # Simulate popping from stack: remove elements smaller than current height
        def pop_smaller(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return pop_smaller(s[:-1])
        
        current_stack = pop_smaller(stack)
        
        # L[idx] is the index of the top of the stack (or -1 if empty)
        # Note: Using 0-indexing, so L[idx] is the index.
        left_boundary = current_stack[-1] if current_stack else -1
        
        # The building at idx is "visible" for i in range(left_boundary + 1, idx)
        # But the problem asks for count c_i for each i.
        # Let's use a difference array to mark ranges.
        # For each j, it adds 1 to c_i for i in [left_boundary + 1, j - 1]
        # Wait, the condition is i < j. 
        # If L[j] is the first index to the left > H[j],
        # then for all i from L[j] to j-1, building j is visible.
        # Example: H = [2, 1, 4], j=2 (H=4). L[2]=-1. i=0,1 are valid.
        # Example: H = [2, 4, 3], j=2 (H=3). L[2]=1 (H=4). i=1 is valid.
        
        # We will store the boundaries and process them at the end.
        results.append((left_boundary, idx))
        return find_L(idx + 1, current_stack + [idx], results)

    # To avoid recursion depth issues with the stack slicing, 
    # we use a helper to find L values.
    # Since we can't use loops, we use a recursive function to build the L array.
    def get_left_boundaries(idx, stack, L):
        if idx == N:
            return L
        
        # Find the first element in stack > H[idx]
        # Using a helper to simulate the while loop
        def find_nearest(s):
            if not s: return -1
            if H[s[-1]] > H[idx]: return s[-1]
            return find_nearest(s[:-1])
        
        # We need to pass the pruned stack to the next call
        def prune(s):
            if not s or H[s[-1]] > H[idx]: return s
            return prune(s[:-1])
            
        L.append(find_nearest(stack))
        return get_left_boundaries(idx + 1, prune(stack) + [idx], L)

    # The above is still O(N^2) in worst case due to slicing.
    # Let's use a more efficient approach.
    # For each i, c_i is the number of j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements that would be added to a 
    # monotonic increasing stack starting from i+1.
    
    # Another observation: c_i = (number of j > i such that H[j] is a 
    # prefix maximum of the sequence H[i+1...N]).
    
    # This is a known problem. The answer for i is the number of 
    # nodes in the right-spine of the Cartesian tree rooted at the 
    # maximum element of H[i+1...N].
    # Actually, the simplest way: c_i is the number of elements 
    # in the monotonic stack after processing H[i+1...N].
    
    # Let's use the property: c_i = 1 + c_{next_greater_element(i+1)}
    # if we define it carefully.
    # Let NG[i] be the index of the first building j > i such that H[j] > H[i].
    # Then c_i = (NG[i] - i - 1) + 1 + c_{NG[i]} 
    # Wait, that's not correct.
    
    # Correct logic:
    # For a fixed i, the buildings j that satisfy the condition are:
    # j1 = i + 1
    # j2 = first building after j1 taller than H[j1]
    # j3 = first building after j2 taller than H[j2]
    # ... and so on.
    
    # Let NG[i] be the index of the first building j > i such that H[j] > H[i].
    # Then c_i = 1 + c_{NG[i+1]} if i+1 < N, but only if we consider 
    # the sequence starting from i+1.
    # Let f(i) be the number of visible buildings in the range [i, N-1].
    # The first visible building is always i.
    # The next visible building is NG[i].
    # So f(i) = 1 + f(NG[i]) if NG[i] exists, else 1.
    # The answer for building i (1-indexed) is f(i+1).
    
    # To find NG[i] without loops:
    # We can use recursion to simulate the monotonic stack.
    def compute_ng(idx, stack, ng):
        if idx < 0:
            return ng
        
        # While stack and H[stack[-1]] < H[idx]: stack.pop()
        def pop_smaller(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            return pop_smaller(s[:-1])
        
        # This is still slow. Let's use a different approach for NG.
        # We can find NG by iterating backwards.
        pass

    # Since I cannot use loops, I will use a recursive function to 
    # compute NG and then another to compute f.
    # To avoid O(N^2) slicing, I'll use a list and an index.
    
    # Using a list as a stack and passing the current top index.
    # But I can't mutate the list in a loop. 
    # However, I can use a list and mutate it inside the recursion.
    # The constraint says "no for/while loops", but doesn't forbid 
    # mutating a list passed as an argument.
    
    ng = [N] * N
    stack = []
    
    def build_ng(idx):
        if idx < 0:
            return
        while stack and H[stack[-1]] < H[idx]:
            stack.pop()
        if stack:
            ng[idx] = stack[-1]
        stack.append(idx)
        build_ng(idx - 1)

    # Wait, the prompt says "no for/while loops". 
    # I used 'while' in build_ng. I must replace it.
    
    def build_ng_no_loops(idx, current_stack):
        if idx < 0:
            return
        
        def prune(s):
            if not s or H[s[-1]] > H[idx]:
                return s
            s.pop()
            return prune(s)
        
        pruned = prune(current_stack)
        if pruned:
            ng[idx] = pruned[-1]
        pruned.append(idx)
        build_ng_no_loops(idx - 1, pruned)

    # To avoid recursion depth and slicing, I'll use a trick with map/reduce
    # or a recursive function with a shared list.
    
    # Let's use the property: f(i) = 1 + (f(ng[i]) if ng[i] < N else 0)
    # We can compute f(i) for i = N-1 down to 0.
    
    f = [0] * (N + 1)
    def compute_f(i):
        if i >= N:
            return
        f[i] = 1 + (f[ng[i]] if ng[i] < N else 0)
        compute_f(i + 1)

    # To find NG without loops or recursion depth issues:
    # We can use a divide and conquer approach.
    def find_ng_dc(l, r, ng_arr):
        if l >= r:
            return
        mid = (l + r) // 2
        find_ng_dc(l, mid, ng_arr)
        find_ng_dc(mid + 1, r, ng_arr)
        
        # Merge step: for i in [l, mid], find first j in [mid+1, r] s.t. H[j] > H[i]
        # This is still complex. 
        
    # Let's go back to the recursive stack but use a helper to handle the "while"
    # and increase recursion limit.
    
    # Final attempt strategy:
    # 1. Use a recursive function to find NG (Next Greater Element)
    # 2. Use a recursive function to find f(i)
    # 3. Use map/join for output.
    
    # To prevent RecursionError and avoid loops, I'll use a 
    # recursive function for the stack and sys.setrecursionlimit.
    
    # Re-implementing build_ng without while:
    def run_ng(idx, s):
        if idx < 0: return
        def pop_s(st):
            if not st or H[st[-1]] > H[idx]: return st
            st.pop()
            return pop_s(st)
        
        pop_s(s)
        if s: ng[idx] = s[-1]
        s.append(idx)
        run_ng(idx - 1, s)

    # The problem is that for N=2e5, run_ng will hit recursion limit.
    # But the prompt forbids loops. The only way to process 2e5 
    # without loops is recursion or high-level functions (map, filter, reduce).
    # Python's recursion limit is a bottleneck. 
    # However, I can use a trampoline or just set it very high.
    
    # Let's use the recursive logic.
    
    # Note: The logic for c_i is f(i+1).
    # For i = N-1 (last building), c_N = 0.
    # For i = 0, c_1 = f(1).
    
    # Since I cannot use loops, I will use a recursive function 
    # to simulate the process.
    
    # To