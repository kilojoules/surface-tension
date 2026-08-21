The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional style. While idiomatic Python often uses `for` loops for readability, utilizing high-order functions and comprehensions can lead to more concise, declarative code that describes *what* to calculate rather than *how* to step through the process. For this specific problem, which requires maintaining a state (a monotonic stack) while iterating through a sequence, `functools.reduce` is the ideal tool to simulate a loop by passing the state (the stack and the results list) from one element to the next.

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
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[j] is a new maximum encountered while scanning from i+1 to N.
    # However, the problem asks for this for ALL i.
    # This is equivalent to counting how many j > i have H[j] > max(H[i+1...j-1]).
    # A more efficient way to think about this: 
    # For a fixed j, it contributes to the count of i if H[j] is the maximum 
    # of the range [i+1, j].
    # This is a classic monotonic stack problem. 
    # We process from right to left. For a building i, the buildings j that satisfy
    # the condition are those that would remain on a monotonic decreasing stack
    # (scanning from i+1 to N).
    
    # We use reduce to simulate a loop over the indices in reverse.
    # State: (stack, results)
    # stack: stores heights of buildings to the right that could be "visible"
    # results: stores the count for each i
    
    # To solve this: for a fixed i, the valid j's are the elements of the 
    # monotonic stack maintained by processing elements from i+1 to N.
    # Actually, the simplest way to view this is:
    # For a fixed i, we want to count j > i such that H[j] > max(H[k] for i < k < j).
    # This means j is a "right-side" visible building.
    # If we process from right to left, for index i, the valid j's are 
    # exactly the elements of a monotonic stack of heights encountered so far
    # that are processed by popping elements smaller than the current H[i].
    # Wait, the condition is about buildings BETWEEN i and j.
    # Let's re-read: "No building taller than Building j between i and j".
    # This means H[k] <= H[j] for all i < k < j.
    # This is exactly the definition of elements that would be visible 
    # if you look right from i, but the condition is on H[j], not H[i].
    # Correct logic: For a fixed i, j satisfies the condition if H[j] is 
    # a prefix maximum of the sequence H[i+1...N].
    
    # Let's use the property: j satisfies the condition for i if 
    # H[j] > max(H[i+1...j-1]).
    # This means for a fixed j, it is counted for all i < j such that 
    # H[j] is greater than all heights between i and j.
    # This is equivalent to saying i must be greater than the index of the 
    # first building to the left of j that is taller than H[j].
    # Let L[j] be the index of the first building k < j such that H[k] > H[j].
    # Then j satisfies the condition for all i such that L[j] <= i < j.
    # (If no such k exists, L[j] = 0).
    # The number of such i is j - L[j].
    # We need to find the sum of these contributions for each i.
    # This is a range update problem: for each j, add 1 to range [L[j], j-1].
    # We can use a difference array ( Fenwick tree/Segment tree is overkill).
    
    # To find L[j] for all j using a monotonic stack without loops:
    # We use reduce to maintain a stack of (height, index).
    
    # Step 1: Find L[j] for all j
    # stack stores indices of buildings in decreasing order of height
    def find_L(state, curr_idx):
        stack, L = state
        # Pop from stack while H[stack[-1]] < H[curr_//N] ... 
        # Since we can't loop, we use a helper function with recursion 
        # (but recursion is forbidden) or a different approach.
        # Actually, the prompt forbids loops and recursion. 
        # This forces the use of high-order functions.
        # To simulate a 'while' loop for the stack, we can use a nested reduce 
        # or a custom function passed to reduce, but the 'while' is the core.
        # However, we can use a trick: since we need to avoid 'for' and 'while',
        # we can use a recursive-like structure inside a comprehension or 
        # use the fact that we can use 'map' and 'reduce'.
        # But wait, the prompt says "avoid explicit loops and recursion".
        # This is a challenge for a monotonic stack.
        # Let's use a different approach: 
        # For a fixed i, the answer is the number of elements in the 
        # monotonic stack of H[i+1...N].
        # If we process from N down to 1, the answer for i is the size of the 
        # stack after processing H[i+1...N] and removing elements smaller than H[i]?
        # No, the condition is: H[k] <= H[j] for i < k < j.
        # This means j is a "right-side" visible building.
        # The buildings j that satisfy this are exactly the ones that 
        # form a monotonic increasing sequence of heights starting from i+1.
        # Example 1: 2 1 4 3 5
        # i=1: H[2]=1 (ok), H[3]=4 (ok, 4>1), H[4]=3 (no, 4>3), H[5]=5 (ok, 5>4). Total: 3.
        # This is exactly the number of elements in a monotonic stack 
        # when processing H[i+1...N] from left to right, keeping only 
        # elements that are greater than all previous elements.
        # Wait, that's just the count of prefix maximums of H[i+1...N].
        
        # Let's use the L[j] logic. L[j] is the index of the first k < j with H[k] > H[j].
        # We can find L[j] by iterating and maintaining a stack.
        # To avoid 'while', we can use a recursive function, but recursion is forbidden.
        # This leaves us with `reduce` and `map`. 
        # To simulate a `while` loop inside `reduce`, we can use another `reduce` 
        # over a range that is guaranteed to be large enough, but that's hacky.
        # Actually, the most "functional" way to implement a monotonic stack 
        # without loops/recursion is to use `reduce` to manage the stack 
        # and a list comprehension to filter the stack (though that's a loop).
        # But the prompt says "avoid explicit loops" (for/while). 
        # List comprehensions and map/reduce are generally accepted as functional.
        
        return (stack, L)

    # Correct approach using reduce to simulate the monotonic stack:
    # We process from right to left. For index i, the answer is the size of the 
    # monotonic stack of heights to terms i+1...N.
    # When moving from i+1 to i, we add H[i+1] to the stack and remove 
    # all elements smaller than H[i+1] from the TOP of the stack? 
    # No, that's for a different problem.
    
    # Let's use the property: j satisfies the condition for i if 
    # H[j] > max(H[i+1...j-1]).
    # This means j is a "visible" building when looking right from i.
    # The set of such j's are the indices that would remain in a 
    # monotonic increasing stack when processing H[i+1...N] from left to right.
    # This is still hard without loops.
    
    # Let's use the L[j] logic: count = sum_{j=i+1}^N [L[j] <= i]
    # L[j] = max({k < j | H[k] > H[j]} union {0})
    # We can find L[j] using a stack. To avoid 'while', we can use 
    # a recursive function and `sys.setrecursionlimit`. 
    # But recursion is forbidden. 
    # The only way to simulate a while loop in Python without for/while/recursion 
    # is using `reduce` and slicing/filtering.
    
    # To find L[j]:
    # We maintain a stack of indices whose heights are decreasing.
    # For a new j, we need to remove elements from the stack that are smaller than H[j].
    # We can use `reduce` to maintain the stack, and inside, use a 
    # list comprehension or `filter` to simulate the popping.
    # Since we need to remove elements from the END of the stack, 
    # we can use a slice that keeps only the elements that are greater than H[j].
    # But we can't know how many to remove without a loop.
    # Actually, we can use `bisect_right` on a sorted list of heights 
    # if we maintain the stack specifically.
    
    import bisect
    
    # For a fixed i, the answer is the number of j > i such that 
    # H[j] > max(H[i+1...j-1]).
    # This is equivalent to: j is a "record" in the sequence H[i+1...N].
    # Let's process from right to left.
    # When we are at i, we want to count j \in {i+1...N} such that 
    # H[j] is a prefix maximum of H[i+1...N].
    # This is equal to the number of elements in a monotonic stack 
    # maintained by processing H[N...i+1] and keeping elements that 
    # are larger than all elements to their right.
    # No, that's not right.
    
    # Let's go back to L[j]: j is counted for i if L[j] <= i < j.
    # L[j] is the index of the first k < j such that H[k] > H[j].
    # We can find L[j] by using a monotonic stack.
    # To simulate the 'while' loop to pop elements:
    # We can use a recursive function. The prompt says "avoid", 
    # which usually means "try not to", but in strict competitive 
    # programming contexts, it might mean "forbidden".
    # However, if I use `reduce` and a helper function that uses 
    # a list comprehension to find the split point, it works.
    
    def get_L(H):
        # stack stores indices. We want the last index k < j where H[k] > H[j].
        # Since we can't use while, we use reduce to maintain the stack.
        # To find the first element > H[j], we can't easily without a loop.
        # UNLESS we use the fact that the stack is sorted by height.
        # If we keep the stack of (height, index) sorted by height, 
        # we can use bisect. But the stack must be sorted by index.
        # Wait, if we maintain a stack of indices whose heights are 
        # strictly decreasing, then the heights are sorted.
        # We can use bisect_right on the heights in the stack to find 
        # the first height > H[j].
        
        # stack_heights: heights of buildings in the current monotonic stack
        # stack_indices: corresponding indices
        # For H[j], we need the index of the first height in the stack > H[j].
        # Since the stack is decreasing, we search for the first element > H[j] 
        # from the right (end of the stack).
        # This is not a simple bisect because the stack is decreasing.
        # We can store negative heights to use bisect_right.
        
        def step(state, curr):
            idx = curr[0