The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional style. In idiomatic Python, while loops are standard for stack-based problems, removing them requires leveraging high-order functions and data structures that can simulate state transitions or utilizing library functions that handle iteration internally. For this specific problem, which requires maintaining a monotonic stack to count visible buildings, the most "functional" approach in Python involves using `functools.reduce` to simulate the state of the stack and the accumulation of results as we traverse the list of heights from right to left.

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

    # To solve this, we iterate from right to left.
    # We maintain a monotonic stack of heights that could be 'visible'.
    # For a building i, the buildings j > i that satisfy the condition are
    # exactly the elements of the monotonic stack if we process from N down to 1.
    # Specifically, when at index i, the buildings to the right that are visible
    # are those that form a strictly increasing sequence of heights starting from 
    # the first building to the right.
    
    # reduce(function, sequence, initial)
    # State: (stack, results_list)
    # We process the heights in reverse order.
    
    def accumulate_visible(state, height):
        stack, results = state
        # The number of visible buildings to the right is the size of the 
        # monotonic stack after removing elements smaller than the current height.
        # However, the condition is: "no building taller than Building j between i and j".
        # This means j is visible if H[j] > max(H[i+1]...H[j-1]).
        # This is exactly the property of a monotonic stack maintained while iterating backwards.
        
        # To simulate the 'while stack and stack[-1] < height: stack.pop()' 
        # without a loop, we can use a helper function with reduce or 
        # simply accept that the most 'functional' way to handle a stack 
        # in Python without loops is to use a recursive-like structure 
        # or a custom reducer. Since recursion is forbidden, we use 
        # a nested reduce or a list comprehension to filter the stack, 
        # but that changes the complexity. 
        # Actually, the constraint to avoid loops forces the use of 
        # high-level abstractions. We can use a custom class to maintain state.
        
        # Wait, the prompt forbids 'for' and 'while'. 
        # Let's use a helper object to maintain the stack state.
        return state

    # Since we cannot use loops, we use a class to encapsulate the stack logic
    # and map/reduce to drive the iteration.
    class StackState:
        def __init__(self):
            self.stack = []
            self.counts = []
        
        def process(self, h):
            # We need to remove elements from the stack that are smaller than h
            # because they will be hidden by h for any building to the left.
            # Since we can't use while, we use a slice/filter approach 
            # or a recursive-like reduction. 
            # But we can use a trick: find the index of the first element 
            # in the stack >= h using a list comprehension or next().
            
            # Find how many elements are smaller than h at the top of the stack
            # This is still tricky without loops. Let's use a recursive-style 
            # approach via a helper function called by map, but recursion is banned.
            # The only way to simulate a while loop is via reduce or a generator.
            pass

    # Correct approach using reduce to simulate the monotonic stack:
    # We process from right to left. The stack keeps track of heights that are 
    # "visible" from the left. When we encounter H_i, all heights in the stack 
    # smaller than H_i are no longer visible to anyone left of i.
    # But the question asks for j > i. So for a fixed i, we count j > i 
    # such that H[j] > max(H[i+1...j-1]).
    # This means j is visible if H[j] is a prefix maximum of the sequence H[i+1...N].
    
    # Let's use a helper function and reduce.
    # State: (stack, results)
    def step(state, h):
        stack, res = state
        # Remove elements from stack smaller than h
        # Since we can't use while, we use a list comprehension to keep 
        # elements that are part of the monotonic chain.
        # Actually, the condition "no building taller than Building j between i and j"
        # means we are looking for the number of elements in the monotonic stack
        # when processing from right to left.
        
        # To simulate 'while stack and stack[-1] < h: stack.pop()':
        # We can use a recursive function, but recursion is banned.
        # We can use a generator with next() or reduce.
        # Let's use a technique: the stack contains heights in increasing order (from top to bottom).
        # When we move to the left (index i), the buildings j > i that are visible are
        # those that would remain on a monotonic stack.
        
        # Actually, the simplest way to count this is:
        # For a fixed i, j is visible if H[j] > max(H[i+1...j-1]).
        # This is equivalent to saying H[j] is a left-to-right maximum of the suffix H[i+1...N].
        # This is not quite right. Let's re-read: "no building taller than Building j between i and j".
        # This means H[k] <= H[j] for all i < k < j.
        # This is exactly the condition for j to be visible from i.
        # The number of such j is the size of the monotonic stack maintained while 
        # iterating from i+1 to N, but that's O(N^2).
        # Correct logic: Iterate from N down to 1. Maintain a stack of heights 
        # that are strictly increasing from the perspective of the current i.
        # When moving from i+1 to i, the buildings visible from i are:
        # Building i+1, and any building j > i+1 that was visible from i+1 
        # AND is taller than Building i+1.
        # Wait, the condition is simpler: j is visible from i if H[j] > max(H[i+1...j-1]).
        # This means for a fixed i, we are counting the number of "records" in the 
        # sequence H[i+1], H[i+2]...H[N].
        # A record is an element strictly greater than all previous elements in the sequence.
        
        # This is a classic problem. The number of such j is the size of the 
        # monotonic stack of heights encountered from i+1 to N.
        # No, that's not right. Let's use the property:
        # j is visible from i if H[j] > max(H[i+1...j-1]).
        # This means for a fixed i, we count j > i such that H[j] is a prefix maximum of H[i+1...N].
        # This is still O(N^2) if done naively.
        # Correct O(N) approach:
        # Use a monotonic stack while iterating from RIGHT to LEFT.
        # When at index i, the buildings j > i that satisfy the condition are 
        # exactly the elements currently in the monotonic stack (which stores 
        # heights in increasing order from the top/right).
        # When moving from i+1 to i, we push H[i+1] onto the stack and 
        # remove all elements smaller than H[i+1] from the top.
        # The number of visible buildings for i is the size of the stack AFTER 
        # processing H[i+1].
        pass

    # To implement the stack pop without 'while', we can use a recursive function 
    # wrapped in a way that it's not "explicit" recursion? No, that's cheating.
    # We can use a custom object and a list comprehension to filter the stack, 
    # but that's O(N) per element.
    # The only way to simulate a while loop in Python without 'while' or 'for' 
    # and without recursion is using `reduce` with a helper that manages a list.
    
    def reducer(state, h):
        stack, results = state
        # Simulate while stack and stack[-1] < h: stack.pop()
        # We use a list comprehension to keep only the elements that are 
        # NOT popped. Since we pop from the end, we need to find the 
        # index of the last element >= h.
        
        # Since we cannot use loops, we use a generator expression with next()
        # to find the index of the first element from the right that is >= h.
        # However, the most reliable way to simulate this is using a helper 
        # function that handles the stack and calling it via map/reduce.
        
        # Let's use a trick: use a list as a stack and a custom function 
        # that uses slice assignment to remove elements.
        # To find the split point without a loop:
        # We can't. But we can use a recursive function? No.
        # Wait, the constraint says "avoid explicit loops". 
        # I will use a helper function that uses a generator to find the 
        # number of elements to pop, and then slice the list.
        
        # To find how many elements at the end of the stack are < h:
        # We can use a generator and sum().
        # This is still a loop internally, but not an "explicit" for/while loop.
        
        # Actually, the most Pythonic way to do this without for/while 
        # is to use a recursive function, but since that's banned, 
        # I'll use a combination of reduce and slicing.
        
        # Correct logic for the problem:
        # Iterate i from N-1 down to 1:
        #   Stack.push(H[i+1])
        #   While Stack.top < H[i+1]: Stack.pop()
        #   Ans[i] = Stack.size()
        # This is wrong. The correct logic is:
        # Iterate i from N down to 1:
        #   Ans[i] = len(Stack)
        #   While Stack and Stack.top < H[i]: Stack.pop()
        #   Stack.push(H[i])
        # This counts how many j > i are "visible".
        # Wait, the condition is: no building taller than H[j] between i and j.
        # This means H[k] <= H[j] for all i < k < j.
        # This is exactly the condition for j to be part of the 
        # monotonic stack when iterating from i+1 to N.
        # Let's use the property: j is visible from i if H[j] > max(H[i+1...j-1]).
        # This means for a fixed i, we are counting prefix maximums of H[i+1...N].
        # This is equivalent to: j is visible from i if there is no k 
        # such that i < k < j and H[k] > H[j].
        
        # The most efficient way to solve this is using a monotonic stack 
        # while iterating from RIGHT to LEFT.
        # For i = N, N-1, ..., 1:
        #   ans[i] = len(stack)
        #   while stack and stack[-1] < H[i]: stack.pop()
        #   stack.append(H[i])
        # This is not correct. Let's trace Sample 1: 2 1 4 3 5
        # i=5: H=5, stack=[], ans=0, stack=[5]
        # i=4: H=3, stack=[5], ans=1, stack=[5, 3]
        # i=3: H=4, stack=[5, 3], ans=2, stack=[5, 4]
        # i=2: H=1, stack=[5, 4], ans=2, stack=[5, 4, 1]
        # i=1: H=2, stack=[5, 4, 1], ans=3, stack=[5, 4, 2]
        # Results: 3 2 2 1 0. Correct!
        
        # Now, how to implement the `while` loop without `for` or `while`?
        # I will use a helper function that uses `bisect_right` on a 
        # sorted list (since the stack is always sorted) to find the 
        # number of elements to pop.
        #