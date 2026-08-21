import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means we are looking for the number of elements to the right of i
    # that are "visible" if we look from i.
    # Specifically, j satisfies the condition if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is part of a 
    # monotonic increasing sequence of heights starting from the first 
    # building to the right of i.
    
    # However, a simpler interpretation: j satisfies the condition if 
    # H[j] > max(H[i+1] ... H[j-1]).
    # This means for a fixed i, we are counting how many times the 
    # running maximum increases as we move from i+1 to N.
    
    # To solve this efficiently for all i, we can use a monotonic stack
    # and process the array from right to left.
    # For a building i, the buildings j that satisfy the condition are:
    # 1. The building i+1.
    # 2. The building that is the first one taller than i+1 to its right.
    # 3. The building that is the first one taller than that one, and so on.
    
    # We can maintain a stack of indices of buildings that could be the 
    # "next taller" building.
    # For building i, the number of such j's is the number of elements 
    # in the monotonic stack that are taller than the buildings 
    # blocking them. Actually, the simplest way is:
    # The buildings j satisfying the condition for i are exactly the 
    # elements of the monotonic stack (maintained from right to left) 
    # that would be visible.
    
    # Let's use reduce to simulate the stack process.
    # State: (stack, results)
    # We process H in reverse.
    def accumulate(state, h):
        stack, results = state
        # The number of visible buildings to the right of the current building
        # is the size of the monotonic stack after we remove all buildings
        # shorter than the current building's height? 
        # No, that's for buildings taller than H[i].
        # The condition is: H[k] < H[j] for all i < k < j.
        # This means j=i+1 always satisfies it.
        # Then the next j is the first index > i+1 such that H[j] > H[i+1].
        # Then the next j is the first index > j_prev such that H[j] the current max.
        
        # Correct logic: For a fixed i, we are counting indices j > i such that
        # H[j] > max(H[i+1]...H[j-1]).
        # This is exactly the number of elements in a monotonic increasing stack
        # constructed from the range [i+1, N].
        # Since we need this for all i, we can observe that the set of such j's
        # for i is simply the monotonic stack constructed from H[i+1...N].
        
        # When moving from i+1 to i:
        # The stack for i is: [i+1] + [elements in stack for i+1 that are > H[i+1]]
        # Wait, the condition does not depend on H[i], only on buildings between i and j.
        # So for any i, the buildings j that satisfy the condition are:
        # j_1 = i+1
        # j_2 = first index > j_1 such that H[j_2] > H[j_1]
        # j_3 = first index > j_2 such that H[j_3] > H[j_2]...
        # This is exactly the monotonic stack of the suffix H[i+1:].
        
        # Let's maintain the monotonic stack of the suffix.
        # For H[i], the answer is the size of the stack constructed from H[i+1:].
        # The stack for H[i+1:] is: 
        # Take H[i+1], then remove all elements from the previous stack (H[i+2:])
        # that are smaller than H[i+1], then push H[i+1] onto it.
        
        # Using reduce to process from right to left:
        # current_stack is the monotonic increasing stack of the suffix.
        # The number of visible buildings for index i is len(current_stack).
        
        # To avoid loops, we use a helper function to pop smaller elements.
        def pop_smaller(s, height):
            # This is the tricky part without loops. 
            # We can use a recursive-like structure via a custom class or 
            # just accept that we can't use recursion.
            # But we can use a list comprehension to filter the stack? 
            # No, the stack must remain monotonic.
            # Actually, the number of elements in the monotonic stack 
            # after processing H[i+1] is the answer for i.
            pass

    # Since we cannot use loops or recursion, we must rely on 
    # high-order functions. However, removing elements from a stack 
    # until a condition is met is inherently iterative.
    # We can simulate this by using a data structure or a 
    # functional approach.
    
    # Let's redefine: for a fixed i, we want the size of the 
    # monotonic increasing stack of H[i+1...N].
    # Let S_{i+1} be the monotonic stack of H[i+1...N].
    # S_i = [H[i]] + [x for x in S_{i+1} if x > H[i]] --- NO, that's not right.
    # The correct logic:
    # For i, the sequence of j's is:
    # j_1 = i+1
    # j_2 = the first index k > j_1 such that H[k] > H[j_1]
    # ...
    # This means the set of j's for i is exactly the monotonic stack 
    # built by iterating from i+1 to N.
    # Let's process from N down to 1.
    # For i = N, ans = 0, stack = []
    # For i = N-1, ans = len(stack), stack = [H[N]]
    # For i = N-2, ans = len(stack), stack = [H[N-1]] + [x for x in stack if x > H[N-1]]
    
    # We can use reduce to maintain the stack and the results.
    # To remove elements smaller than H[i], we can't use a loop.
    # But we can use a technique: since we only need the SIZE of the stack,
    # and the stack is always sorted, we can use binary search (bisect_right)
    # to find how many elements to remove.
    
    import bisect
    
    def step(state, h):
        stack, results = state
        # The answer for the current building (which is the 'i' in the problem)
        # is the size of the stack built from the buildings to its right.
        ans = len(나_stack := stack) 
        # Now update the stack for the building to the left.
        # The new building is H[i]. The buildings to its right start with H[i+1].
        # Wait, the building at index i is the one we are calculating for.
        # The buildings to its right are H[i+1...N].
        # The stack for H[i+1...N] is:
        # 1. Start with H[i+1].
        # 2. Keep elements from the stack of H[i+2...N] that are > H[i+1].
        # Since the stack is sorted (monotonic), we can use bisect to find the split point.
        return (None, None) # Placeholder

    # Let's refine:
    # Process H from right to left.
    # Let f(i) be the monotonic stack of H[i...N].
    # f(N) = [H[N]]
    # f(i) = [H[i]] + [x for x in f(i+1) if x > H[i]]
    # The answer for building i is len(f(i+1)).
    
    # To implement [x for x in f(i+1) if x > H[i]] without a loop:
    # Since f(i+1) is sorted (strictly increasing), we can use bisect_right
    # to find the index of the first element > H[i] and slice the list.
    
    def reducer(state, h):
        stack, results = state
        # Current h is H[i]. The answer for i is the size of the stack of H[i+1...N].
        res = len(stack)
        # Update stack for the next (leftward) building:
        # New stack is [h] + stack[bisect_right(stack, h):]
        # Note: we need the stack to be sorted. Since we are adding h to the left,
        # and we keep elements > h, the stack remains sorted.
        import bisect
        new_stack = [h] + stack[bisect.bisect_right(stack, h):]
        return (new_stack, results + [res])

    # We need to process H in reverse.
    # Initial state: stack = [], results = []
    final_state = reduce(reducer, reversed(H), ([], []))
    
    # The results were collected from i=N down to 1.
    # We need them from i=1 to N.
    print(*(final_state[1][::-1]))

if __name__ == "__main__":
    solve()