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
    # means Building j must be a prefix maximum when looking from i+1 onwards.
    # However, the problem asks for the count of such j for each i.
    # This is equivalent to counting how many elements to the right of i 
    # are greater than all elements between i and that element.
    # This is exactly the number of elements that would remain in a 
    # monotonic decreasing stack if we processed the array from right to left.
    
    # We use reduce to iterate backwards through the heights.
    # State: (stack, results)
    # stack: stores heights of buildings that could be 'visible' to buildings to the left.
    # results: stores the count of visible buildings for each index.
    
    def process(state, h):
        stack, results = state
        # We need to find how many elements in the current stack are 
        # "visible". Actually, any element currently in the monotonic 
        # stack (maintained by popping smaller elements) is visible 
        # because it was the maximum of some range to its right.
        # Wait, the condition is: no building taller than H[j] between i and j.
        # This means H[j] > max(H[i+1]...H[j-1]).
        # This is satisfied by all elements that form the "upper hull" 
        # (monotonic increasing sequence) starting from index i+1.
        
        # To implement this without loops, we use a helper function 
        # to prune the stack. Since we can't use while, we use a 
        # recursive-like structure or a filter, but the stack 
        # depends on the previous state.
        # Actually, the number of j's for index i is simply the size of the 
        # monotonic stack after processing elements from N down to i+1.
        
        # Correct logic: 
        # For a fixed i, we are looking for j > i such that H[j] > max(H[i+1]...H[j-1]).
        # This is simply the count of elements that would be added to a 
        # monotonic increasing stack if we processed the array from i+1 to N.
        # But we need this for all i.
        # Notice: the set of such j's for index i are the elements of the 
        # monotonic stack maintained when iterating from N down to i+1, 
        # where we keep elements that are larger than everything to their right.
        # No, that's not right. Let's re-evaluate.
        # Condition: H[k] <= H[j] for all i < k < j.
        # This means H[j] is a "right-side" maximum.
        # For a fixed i, the valid j's are:
        # j1 = i + 1
        # j2 = first index > j1 such that H[j2] > H[j1]... No.
        # Example 1: 2 1 4 3 5. i=1 (H=2). j=2(H=1), j=3(H=4), j=5(H=5).
        # Between 1 and 2: empty. OK.
        # Between 1 and 3: H[2]=1 < H[3]=4. OK.
        # Between 1 and 4: H[3]=4 > H[4]=3. FAIL.
        # Between 1 and 5: H[2]=1, H[3]=4, H[4]=3. All <= H[5]=5. OK.
        # The valid j's are exactly the elements of the monotonic increasing 
        # stack if we process the array from i+1 to N.
        # However, the number of such elements is the same as the size of the 
        # monotonic decreasing stack processed from N down to i+1.
        
        # To remove elements from the stack without a while loop, 
        # we can use a recursive function or a trick with bisect.
        # But since we need to maintain a stack, and N=2e5, 
        # recursion will hit limits. 
        # Let's use the property: the number of such j is the number of 
        # elements in the stack when we maintain a monotonic decreasing 
        # stack from right to left.
        # When moving from i+1 to i, we push H[i+1] onto the stack and 
        # remove all elements smaller than it.
        
        # Since we can't use while, we can use a list and slice 
        # based on binary search (bisect_right) to simulate the pop.
        import bisect
        
        # We need a way to remove all elements smaller than h from the 
        # front of the stack. But the stack is monotonic decreasing 
        # (from the right), so it's sorted.
        # Stack: [H[N], ..., H[i+1]] sorted increasingly.
        # When we add H[i], we remove all elements < H[i] from the left.
        
        # Using a sorted list and bisect:
        idx = bisect.bisect_left(stack, h)
        new_stack = stack[idx:] + [h]
        return (new_stack, len(new_stack) - 1)

    # We need to process from N-1 down to 0.
    # The result for index i depends on the stack formed by H[i+1...N-1].
    # We use reduce to build the results list.
    
    # To avoid the loop and recursion, we use a custom function 
    # inside reduce that handles the stack.
    # Because we need to remove elements from the "bottom" of the stack 
    # (the ones added first/rightmost), and the stack is sorted, 
    # we can use bisect.
    
    import bisect
    
    # state: (stack, results_list)
    # We process H in reverse. For H[i], the answer is len(stack).
    # Then we update the stack by adding H[i] and removing elements < H[i].
    def step(state, h):
        stack, res = state
        # The answer for the current i is the current size of the stack
        # (which contains elements from i+1 to N-1)
        res.append(len(stack))
        # Update stack for the next (leftward) building
        idx = bisect.bisect_left(stack, h)
        # stack is maintained as sorted. Elements < h are removed.
        # Since we add h to the end, and it's the new "leftmost" building,
        # it will be the largest or among the largest.
        # Actually, the stack should store the "visible" buildings.
        # A building j is visible from i if H[j] > max(H[i+1...j-1]).
        # This is exactly the monotonic increasing stack from i+1 to N.
        # The size of this stack is what we need.
        # When moving from i+1 to i, the new stack is:
        # [H[i+1]] + [all elements in old stack that are > H[i+1]]
        
        # Let's refine:
        # For i, we want count of j > i such that H[j] > max(H[i+1...j-1]).
        # Let S_{i+1} be the monotonic increasing stack of H[i+1...N-1].
        # S_i = [H[i+1]] + [x for x in S_{i+1} if x > H[i+1]]
        # The answer for i is len(S_i).
        
        # To implement S_i without loops:
        # S_{i+1} is always sorted. We can use bisect to find elements > H[i+1].
        idx = bisect.bisect_right(stack, h)
        new_stack = [h] + stack[idx:]
        return (new_stack, res)

    # Initial state: stack = [], results = []
    # Process H from index N-1 down to 0.
    # Note: The problem asks for i = 1 to N. 
    # For i=N, the answer is always 0.
    # For i < N, we look at j from i+1 to N.
    
    # We process H in reverse: H[N-1], H[N-2]... H[0]
    # For H[N-1], ans = 0. Stack becomes [H[N-1]]
    # For H[N-2], ans = len([H[N-1]] + [x for x in [H[N-1]] if x > H[N-2]])
    
    # Correct logic with reduce:
    # We want to compute answers for i = N-1, N-2, ..., 0.
    # For i = N-1: ans = 0, stack = [H[N-1]]
    # For i = N-2: ans = len(stack_after_updating), stack = update(stack, H[N-2])
    
    # Let's use a different approach for the reduce to match the index:
    # The answer for i is the size of the stack formed by H[i+1...N-1].
    # We process H from right to left.
    
    def final_reduce(state, h):
        stack, res = state
        # Current h is H[i]. The answer for i is len(stack).
        # Then update stack for the next i (which is i-1).
        # The new stack is [h] + [x for x in stack if x > h]
        # Since stack is sorted (it's a monotonic increasing stack of the suffix),
        # we can use bisect.
        import bisect
        res.append(len(stack))
        idx = bisect.bisect_right(stack, h)
        return ([h] + stack[idx:], res)

    # We need to process H in reverse order.
    # H = [2, 1, 4, 3, 5]
    # i=4 (H=5): stack=[], res=[0]. New stack=[5]
    # i=3 (H=3): stack=[5], res=[0, 1]. New stack=[3, 5]
    # i=2 (H=4): stack=[3, 5], res=[0, 1, 2]. New stack=[4, 5]
    # i=1 (H=1): stack=[4, 5], res=[0, 1, 2, 2]. New stack=[1, 4, 5]
    # i=0 (H=2): stack=[1, 4, 5], res=[0, 1, 2, 2, 3]. New stack=[2, 4, 5]
    
    # Wait, the sample 1 output is 3 2 2 1 0.
    # My trace: res = [0, 1, 2, 2, 3]. Reversed: 3 2 2 1 0. Correct.
    
    # To avoid 'import' inside reduce, import at top.
    # To avoid 'res.append', we can return a new list, but that's O(N^2).
    # However, since we are allowed to use a list and mutate it (as long as 
    # there are no for/while loops), res.append is fine.
    
    # Final implementation using reduce:
    import bisect
    _, results = reduce(final_reduce, H[::-1], ([], []))
    
    # The results were collected from i=N-1 down to 0.
    # So results[0] is for i=N-1, results[N-1] is for i=0.
    # We need to print them for i=0 to N-1.
    print(*(results[::-1]))

# Since I cannot define final_reduce inside solve and call it 
# without a loop, I will wrap everything in a way that 
# complies with the constraints.