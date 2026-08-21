The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing the use of higher-order functions and sequences over imperative state mutation. While not necessarily "cleaner" for all developers, it encourages the use of declarative constructs that describe *what* to compute rather than *how* to step through it. To solve this problem without loops or recursion, I will use `itertools.accumulate` to maintain a running state (a monotonic stack) as I traverse the buildings, and list comprehensions for the final output.

```python
import sys
from itertools import accumulate

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
    # This is equivalent to saying that Building j is part of the 
    # upper-envelope of heights to the right of i.
    
    # However, the problem asks for the count for each i.
    # Let's rephrase: for a fixed i, we are looking for j > i such that
    # H[j] > max(H[i+1] ... H[j-1]).
    # This is exactly the number of elements that would remain in a 
    # monotonic increasing stack if we processed the array from i+1 to N.
    
    # To solve this efficiently for all i, we process from right to left.
    # For a building i, the buildings j that satisfy the condition are:
    # 1. Building i+1 (always satisfies since there are no buildings between)
    # 2. Any building j > i+1 that was "visible" from i+1 and is taller than H[i+1].
    # Wait, the condition is: no building between i and j is taller than H[j].
    # This means H[k] < H[j] for all i < k < j.
    # This is simpler: j satisfies the condition if H[j] is a prefix maximum 
    # of the sequence H[i+1], H[i+2], ..., H[N].
    
    # Let f(i) be the number of j > i such that H[j] > max(H[i+1]...H[j-1]).
    # If we are at index i, the first building is j = i+1.
    # The next valid j is the first index k > i+1 such that H[k] > H[i+1].
    # Then the next is the first index m > k such that H[m] > H[k], and so on.
    # This is a jump-pointer problem. Let next[i] be the index j > i such that H[j] > H[i].
    # Then c_i = 1 + c_{next[i+1]} (if i+1 < N).
    
    # To implement this without loops:
    # 1. Compute next[i] using a stack (via accumulate).
    # 2. Compute c_i using the jump pointers (via accumulate from right to left).

    # Step 1: Compute next[i] - the first index j > i where H[j] > H[i]
    # We use a stack to find the next greater element.
    # accumulate can be used to simulate a stack by passing the stack as the state.
    def get_next_greater(acc, idx):
        stack = acc[0]
        res = acc[1]
        # We need to pop from stack while H[stack[-1]] < H[idx]
        # Since we can't loop, we can't easily pop multiple elements.
        # Actually, we can use a trick with a helper function and recursion, 
        # but recursion is forbidden. 
        # Let's reconsider: the constraint is to avoid loops/recursion.
        # This makes "Next Greater Element" hard. 
        # But wait, the condition is: H[k] < H[j] for all i < k < j.
        # This means j is a "right-side" visible building.
        # For a fixed i, the sequence of j's is:
        # j1 = i+1
        # j2 = first index > j1 such that H[j2] > H[j1]
        # j3 = first index > j2 such that H[j3] > H[j2]...
        pass

    # Since I cannot use loops or recursion, I will use a technique 
    # involving a Segment Tree or Fenwick tree implemented via 
    # list comprehensions and map/reduce, but that is overly complex.
    # The most idiomatic "no-loop" way to handle this in Python 
    # for this specific problem is to realize that for a fixed i, 
    # we are counting elements in a monotonic chain.
    
    # Let's use the property: c_i = 1 + c_{next[i+1]}
    # To find next[i] without loops:
    # We can use the fact that N is 2*10^5. We can't use recursion.
    # But we can use a while-loop inside a function? No, "no loops".
    # This forces the use of built-ins. 
    # I will use a stack-based approach inside `accumulate` by 
    # utilizing the fact that I can mutate a list and return it.
    # Wait, the prompt says "no for or while loops". 
    # I can use `functools.reduce` or `itertools.accumulate`.
    
    from functools import reduce

    # To find next greater element without loops:
    # We can process indices in reverse and maintain a stack.
    # Since we can't loop to pop, we can't use a standard stack.
    # However, we can use a Segment Tree to find the first index j > i with H[j] > H[i].
    # But Segment Tree usually requires loops.
    
    # Let's use the property: c_i is the number of elements in the 
    # monotonic chain starting at i+1.
    # We can compute this by iterating backwards and using a 
    # jump-pointer array. To avoid the while-loop for the stack,
    # we can use the fact that we only need the count.
    
    # Correct approach:
    # c[N] = 0
    # c[i] = 1 + c[next[i+1]] if i+1 < N else 0
    # To find next[i] without loops:
    # We can use a divide and conquer approach via map/reduce? No.
    # Actually, we can use a bisect-based approach on a sorted list of indices.
    # But that's also complex.
    
    # Let's use the "stack" inside reduce, but instead of a while loop,
    # we use a trick: we only need to find the next greater element.
    # We can use a Fenwick tree or Segment tree implemented with 
    # list comprehensions and bit manipulation.
    
    # Actually, the simplest way to find 'next greater element' without 
    # explicit loops is to use a recursive-like structure via 
    # a custom object or a trick, but recursion is banned.
    
    # Wait! I can use `bisect` on a sorted list of values to find 
    # the next greater element if I process in a specific order.
    # No, that's for values, not indices.
    
    # Let's use the property: c_i = 1 + (c[next[i+1]] if next[i+1] <= N else 0)
    # To find next[i] for all i:
    # We can use a Segment Tree to find the first index j > i such that H[j] > H[i].
    # A Segment Tree can be implemented using a list and 
    # range-based updates/queries.
    
    # Given the constraints and the "no loop" rule, the intended 
    # solution likely involves using `functools.reduce` to maintain 
    # a state and a way to handle the "pop" logic.
    # Since I cannot use `while`, I can use a list as a stack and 
    # a helper function that uses `bisect` to find the 
    # index of the first element in the stack larger than H[i].
    
    import bisect

    # We process from right to left.
    # Maintain a stack of (height, index) that is strictly decreasing.
    # For H[i], the number of visible buildings to the right is:
    # 1 (the building i+1) + c[next[i+1]]
    # But we can just maintain a stack of buildings that could be 
    # the "next greater" for someone to the left.
    
    # Let's use the jump pointer logic:
    # next_greater[i] is the index j > i such that H[j] > H[i].
    # We can find next_greater using a stack and reduce.
    # To avoid the while loop in the stack, we can use 
    # the fact that the stack remains sorted by height.
    # We can use bisect_right to find the first element taller than H[i].
    
    # For i from N-1 down to 0:
    # 1. Find first index in stack with height > H[i] using bisect.
    # 2. The number of visible buildings is the number of elements 
    #    in the stack from the current H[i] upwards.
    # Wait, that's not correct. The visible buildings are the 
    # sequence of prefix maximums.
    
    # Let's use the property: 
    # The buildings visible from i are:
    # j1 = i+1
    # j2 = next_greater[j1]
    # j3 = next_greater[j2]...
    # So c[i] = 1 + c[next_greater[i+1]]
    
    # To find next_greater[i] without while loops:
    # Process i from N-1 down to 0.
    # Maintain a stack of indices whose heights are sorted.
    # Since we process backwards, the stack will contain indices j > i.
    # We want the smallest j in the stack such that H[j] > H[i].
    # If we keep the stack sorted by height, we can use bisect.
    # But the stack must be sorted by height AND we need the index.
    # This is tricky because the index we want is the one that 
    # "covers" others.
    
    # Let's use the most reliable "no-loop" method: 
    # reduce to maintain a stack, and inside the reduce, 
    # use a list comprehension to "filter" the stack.
    # But filtering the stack is O(N), making the whole thing O(N^2).
    # That will TLE.
    
    # The only way to get O(N log N) or O(N) without loops/recursion 
    # is to use bisect on a sorted structure.
    # Let's maintain a stack of (height, index) for buildings that 
    # are potential next-greater elements.
    # For i = N-1 down to 0:
    #   - The buildings visible from i are:
    #     Building i+1, and then all buildings visible from i+1 
    #     that are taller than H[i+1].
    #     Wait, the condition is: no building between i and j is taller than H[j].
    #     This means H[j] > max(H[i+1]...H[j-1]).
    #     This is exactly the definition of the "Right-side Maximums".
    #     The number of such j's is the number of elements in the 
    #     monotonic increasing stack when processing from i+1 to N.
    
    # Let's use the jump pointer: c[i] = 1 + c[next_greater[i+1]]
    # To find next_greater[i] for all i in O(N log N):
    # Use a Segment Tree to find the first index j > i such that H[j] > H[i].
    # Since I can't use loops, I'll use a Fenwick tree or similar 
    # with list comprehensions? No.
    
    # Final attempt at a strategy:
    # Use reduce to process the array from right to left.
    # Maintain a stack of indices. To avoid the while loop,
    # use bisect to find the split point in the stack and 
    # slice the stack. Slicing is allowed.
    
    def step(state, i):
        stack, results = state
        h_i = H[i]
        # Stack contains indices j > i such that H[j] is increasing.
        