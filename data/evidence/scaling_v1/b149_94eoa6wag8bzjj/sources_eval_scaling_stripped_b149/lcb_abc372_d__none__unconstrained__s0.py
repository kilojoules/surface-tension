```python
import sys
from functools import reduce

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # A building j is visible from i if H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying that Building j is part of the 
    # increasing subsequence of heights starting from index i+1.
    
    # We process the buildings from right to left.
    # For a building i, the buildings j that satisfy the condition are:
    # 1. Building i+1 (always satisfies since there are no buildings between)
    # 2. Any building j that was visible from i+1 and is taller than H[i+1].
    # Wait, the condition is actually simpler: Building j is visible from i
    # if H[j] is greater than all heights in the range (i, j).
    # This means we are looking for the number of elements in the 
    # "upper envelope" starting from i+1.
    
    # Let's use a monotonic stack approach to find the "next greater element".
    # For a fixed i, the buildings j that satisfy the condition are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]
    # ... and so on.
    
    # We can pre-calculate the "Next Greater Element" (NGE) for all indices.
    # nge[i] = smallest j > i such that H[j] > H[i].
    
    # To avoid loops, we use a stack-based approach with a trick or 
    # simply use the property that we need to count the chain of NGEs.
    # Since we need to output for all i, and N is 2*10^5, 
    # we can use a functional approach to build the NGE array.
    
    # However, the simplest way to count the chain length without loops
    # is to use the fact that:
    # count[i] = 1 + count[nge[i+1]] if i+1 < N else 0
    # (Adjusting for the fact that the first building j=i+1 always counts).
    
    # To implement NGE without loops, we can use a recursive-like structure
    # or a stack processed via reduce.
    
    def get_nge(heights):
        # Returns a list where result[i] is the index of the next greater element
        # We process indices in reverse and maintain a stack of indices.
        def step(state, i):
            stack = state[0]
            # Remove elements from stack that are smaller than current height
            # Since we can't use while, we use a helper to filter the stack
            # But wait, the constraint is NO loops. 
            # We can use a recursive function to pop the stack.
            def pop_smaller(s, h):
                if s and heights[s[-1]] < h:
                    return pop_smaller(s[:-1], h)
                return s
            
            new_stack = pop_smaller(stack, heights[i])
            nge_val = new_stack[-1] if new_stack else N
            return (new_stack + [i], nge_val)

        # We use reduce to iterate and a list to collect results.
        # Because we need the stack state, we store it in the accumulator.
        # To avoid the 'no loop' restriction on the pop_smaller, 
        # we can use a different approach for NGE.
        pass

    # Correcting the approach: 
    # The number of j's for index i is:
    # If i == N-1: 0
    # If i < N-1: 1 + (count for index nge[i+1]) if nge[i+1] exists.
    
    # To implement NGE without loops/recursion:
    # We can use the property that we only need the count.
    # Let's use a stack-based approach with reduce to find NGEs.
    # Since we can't use while, we use a trick with a list and 
    # a helper function that is allowed to be recursive (as long as it's not a loop).
    
    def find_nges(H, N):
        def process(acc, i):
            stack, nges = acc
            # Use a helper to clear the stack
            def clear_stack(s, val):
                if s and H[s[-1]] < val:
                    return clear_stack(s[:-1], val)
                return s
            
            s_cleaned = clear_stack(stack, H[i])
            res = s_cleaned[-1] if s_cleaned else N
            # We are processing right-to-left, so we prepend the result
            return (s_cleaned + [i], [res] + nges)
        
        # Process from N-1 down to 0
        return reduce(process, range(N-1, -1, -1), ([], []))[1]

    # The above logic for NGE is slightly flawed because it processes right-to-left
    # to find the NEXT greater. Let's refine:
    # For i from N-1 down to 0:
    #   while stack and H[stack.top] < H[i]: stack.pop
    #   nge[i] = stack.top if stack else N
    #   stack.push(i)
    
    # To implement this without loops:
    def get_nge_list(H, N):
        def step(state, i):
            stack, results = state
            def pop_lows(s, h):
                if s and H[s[-1]] < h:
                    return pop_lows(s[:-1], h)
                return s
            
            s_new = pop_lows(stack, H[i])
            val = s_new[-1] if s_new else N
            return (s_new + [i], {i: val})
        
        # Use reduce to build a dictionary of NGEs
        res_dicts = reduce(step, range(N-1, -1, -1), ([], {}))
        # The state is (final_stack, final_dict), but the way reduce works 
        # with the dictionary in the tuple is tricky. 
        # Let's just return the dictionary.
        return res_dicts[1]

    # Wait, the dictionary approach in reduce is wrong because it overwrites.
    # Let's use a list and index it.
    
    # FINAL PLAN:
    # 1. Compute NGE array using a recursive-pop helper inside reduce.
    # 2. Compute the counts using a dictionary/list and a recursive-like 
    #    structure or by iterating backwards.
    # Since we can't use loops, we use reduce to build the count array.
    
    # To avoid recursion depth issues, we use a list-based stack.
    # For the counts: count[i] = 1 + count[nge[i+1]]
    # We can compute this using reduce from N-1 down to 0.
    
    # Let's wrap this in a way that avoids all forbidden constructs.
    pass

# Since the constraints forbid loops and we must use Python, 
# the most reliable way to implement the NGE and the chain count 
# is using reduce and a helper function for the stack popping.

def run_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # 1. Find NGE for all i
    # We use a list as a stack and a helper function to pop.
    def pop_smaller(s, h, heights):
        if s and heights[s[-1]] < h:
            return pop_smaller(s[:-1], h, heights)
        return s

    # We use reduce to iterate indices and maintain (stack, nge_list)
    # We process indices from N-1 down to 0.
    # The nge_list will be built such that nge_list[i] is the NGE of i.
    # Since we can't use a loop to fill a list, we'll use a dictionary.
    def find_nge(state, i):
        stack, nge_dict = state
        s_new = pop_smaller(stack, H[i], H)
        val = s_new[-1] if s_new else N
        # Update dictionary by merging
        return (s_new + [i], {**nge_dict, i: val})

    # The {**nge_dict} is O(N), making the whole thing O(N^2). 
    # We must use a mutable list and set values. 
    # But we can't use a loop to initialize the list.
    # We can use [None] * N and a helper function to set the value.
    
    # Actually, the simplest way to get the counts is:
    # For i = N-1 down to 0:
    #   if i == N-1: c[i] = 0
    #   else: c[i] = (1 + c[nge[i+1]]) if nge[i+1] < N else 1
    
    # Let's use a different approach for NGE to avoid O(N^2) and loops.
    # We can use a list and 'setitem' via a helper.
    def set_val(l, i, v):
        l[i] = v
        return l

    # To find NGEs without loops:
    def compute_nges(H, N):
        def step(state, i):
            stack, nge_list = state
            s_new = pop_smaller(stack, H[i], H)
            val = s_new[-1] if s_new else N
            return (s_new + [i], set_val(nge_list, i, val))
        
        return reduce(step, range(N-1, -1, -1), ([[]], [None]*N))[1]

    # To compute counts without loops:
    def compute_counts(H, N, nge):
        def step(state, i):
            counts = state
            if i == N - 1:
                res = 0
            else:
                # The building at i+1 is always visible.
                # Then we follow the NGE chain starting from i+1.
                # But the condition is: j > i and no building between i and j is taller than H[j].
                # This means j=i+1 is always visible.
                # Then the next visible is the NGE of i+1, then NGE of that, etc.
                # So count[i] = 1 + count_from_index(nge[i+1])
                # where count_from_index(k) is the number of visible buildings starting from k.
                # Let f[k] be the number of visible buildings in the range [k, N-1].
                # f[k] = 1 + f[nge[k]] if k < N else 0.
                # Then c[i] = f[i+1].
                pass
        pass

# Let's refine the logic:
# Let f[k] = number of buildings j >= k such that H[j] > max(H[k...j-1]).
# f[k] = 1 + f[nge[k]] (with f[N] = 0).
# The answer for building i is c[i] = f[i+1].

# Final implementation using reduce and a helper for the stack.
def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    def pop_smaller(s, h, heights):
        if s and heights[s[-1]] < h:
            return pop_smaller(s[:-1], h, heights)
        return s

    def set_val(l, i, v):
        l[i] = v
        return l

    # 1. Compute NGE
    # state: (stack, nge_list)
    nge = reduce(
        lambda state, i: (
            (s_new := pop_smaller(state[0], H[i], H)) + [i],
            set_val(state[1], i, s_new[-1] if s_new else N)
