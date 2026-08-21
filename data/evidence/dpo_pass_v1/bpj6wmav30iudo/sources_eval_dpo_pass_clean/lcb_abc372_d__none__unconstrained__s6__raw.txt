The constraint to avoid explicit loops and recursion forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state updates. In idiomatic Python, this encourages the use of high-level abstractions and built-in iterators, though for this specific algorithmic problem (which typically requires a stack), it necessitates the use of `functools.reduce` to simulate state accumulation. I will use `functools.reduce` to process the buildings from right to left, maintaining a monotonic stack to count visible buildings.

```python
import sys
from functools import reduce

def solve():
    # Read input using sys.stdin.read().split() to handle all whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "right-side" visible building.
    # For a fixed i, we are looking for j > i such that for all k (i < k < j), H_k < H_j.
    # This is equivalent to saying that if we look from i to the right, 
    # we count how many times we encounter a new maximum height.
    # However, the problem asks for j such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means for a fixed i, the sequence of H_j that satisfy this are the 
    # elements of the monotonic increasing subsequence starting from i+1.
    
    # To solve this efficiently for all i, we process from right to left.
    # We maintain a monotonic stack of heights that could be the "maximum" for indices to the left.
    # For index i, the number of j's is the number of elements in the monotonic stack 
    # that are taller than any element between i and that element.
    # Actually, a simpler observation: for a fixed i, the j's that satisfy the 
    # condition are exactly the indices j > i that form a strictly increasing 
    # subsequence of heights if we only consider those that are taller than 
    # all elements between i and j.
    # This is equivalent to: j is counted if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means for a fixed i, we are counting the number of "prefix maximums" 
    # of the array H[i+1:].
    
    # Let f(i) be the number of prefix maximums in H[i+1:].
    # If H[i+1] is the tallest building in H[i+1:], then f(i) = 1 + f(i+1) 
    # is NOT necessarily true.
    # Correct logic: Let next_taller[i] be the index j > i such that H[j] > H[i].
    # The buildings that satisfy the condition for i are:
    # 1. Building i+1
    # 2. The building that is the first one taller than Building i+1 to its right.
    # 3. The building that is the first one taller than that, and so on.
    # Let dp[i] be the number of such buildings.
    # dp[N] = 0
    # dp[i] = 1 + dp[next_taller[i+1]] if i+1 < N else 0
    
    # To find next_taller without loops, we use reduce to build the array.
    # We process from right to left to find the first element to the right that is taller.
    
    # Use reduce to build the next_taller array and the dp array.
    # state: (stack, dp_results)
    # stack: indices of buildings that could be the next taller
    
    # Since we need to avoid loops, we use a list comprehension to reverse H 
    # and reduce to calculate the counts.
    
    # For a fixed i, the buildings j that satisfy the condition are:
    # j_1 = i + 1
    # j_2 = first index > j_1 such that H[j_2] > H[j_1]
    # j_3 = first index > j_2 such that H[j_3] > H[j_2]...
    # Let dp[j] = 1 + dp[next_taller[j]]
    # The answer for i is dp[i+1].
    
    # To find next_taller[j] for all j:
    # We use reduce over the indices in reverse.
    
    # Part 1: Find next_taller indices
    # we store (current_stack, next_taller_list)
    # we use a list for next_taller and fill it.
    
    # Because of the "no loop" constraint, we must use reduce to simulate the stack.
    # We'll process indices from N-1 down to 0.
    
    # To compute dp[i] = 1 + dp[next_taller[i]], we need next_taller first.
    # Let's combine them: reduce from right to left.
    # state: (stack, dp_array)
    # stack: indices k such that H[k] is monotonically decreasing (from top to bottom)
    
    # For index i:
    # 1. Pop from stack while H[stack_top] < H[i]
    # 2. next_taller[i] = stack_top if stack else N
    # 3. dp[i] = 1 + dp[next_taller[i]] if next_taller[i] < N else 1
    
    # Since we can't use while loops, we use a helper function with reduce 
    # or a recursive-like structure inside reduce. 
    # But wait, we can use a recursive function to pop the stack? No, recursion banned.
    # We can use a list comprehension or filter to "pop" the stack? 
    # No, that's not efficient.
    
    # Actually, the condition "no building taller than Building j between i and j"
    # is simpler: j satisfies this if H[j] > max(H[i+1...j-1]).
    # This means for a fixed i, we are counting the number of elements in the 
    # "upper envelope" starting from i+1.
    
    # Let's use the property: dp[i] = 1 + dp[next_taller[i]]
    # To find next_taller without while loops:
    # We can use a divide and conquer approach via map/reduce? No.
    # We can use the fact that H_i are distinct and in range [1, N].
    
    # Let's use a different approach: 
    # For each j, it is a "valid j" for all i such that max(H_{i+1...j-1}) < H[j].
    # This means i must be such that for all k from i+1 to j-1, H[k] < H[j].
    # This is true for i in range [last_taller[j], j-1], where last_taller[j] 
    # is the index of the first building to the left of j that is taller than H[j].
    
    # Let L[j] be the index of the first building to the left of j with H[L[j]] > H[j].
    # Then building j contributes 1 to c_i for all i from L[j] to j-1.
    # We can use a difference array to count this.
    
    # To find L[j] without loops:
    # Use reduce to maintain a monotonic stack. 
    # To handle the "while" pop without a loop, we can use a recursive-like 
    # structure via a helper function? No, recursion banned.
    # We can use a custom object or a closure to maintain state in reduce.
    
    # Wait, the "no loop" constraint is very strict. 
    # I will use a list comprehension to generate the indices and 
    # reduce to process the stack, using a helper function that 
    # handles the popping via a list slice or filter.
    # Actually, since N=2e5, we must be careful with complexity.
    
    # Let's use the property: j is counted for i if H[j] is a prefix maximum of H[i+1:].
    # This is a known problem solvable by a Segment Tree or similar, 
    # but we can't use loops.
    
    # Let's use the L[j] approach. To find L[j] without while:
    # We can use a functional approach to the monotonic stack.
    # Since we can't use while, we can use a recursive-like 
    # structure implemented via reduce on a range.
    
    # Actually, the most "functional" way to find the next greater element 
    # is to use the fact that we can process the elements in order of their values.
    # But that requires a data structure.
    
    # Let's use a trick: we can simulate a while loop using a 
    # recursive-like call inside a reduce by manipulating the stack.
    # But recursion is banned.
    
    # Let's use the property: L[j] is the index of the first element to the left 
    # that is greater than H[j].
    # We can find this by processing the array and using a 
    # mathematical approach or a built-in that we can leverage.
    
    # Given the constraints and the "no loop" rule, 
    # the only way to maintain a stack is reduce.
    # To "pop" without a while loop, we can use a recursive-like 
    # function? No. 
    # We can use a list comprehension to find the first element in the stack 
    # that is greater than H[j].
    # stack = [index for index in stack if H[index] > H[j]] 
    # This is O(N^2).
    
    # However, we can use a Fenwick tree or Segment tree implemented 
    # via list comprehensions and reduce? No.
    
    # Let's reconsider: dp[i] = 1 + dp[next_taller[i]]
    # We can find next_taller by sorting the buildings by height.
    # For the tallest building, next_taller is N.
    # For others, we need the first index to the right that is taller.
    # We can use a Disjoint Set Union (DSU) to find the next index.
    # But DSU usually requires a while loop for find().
    
    # Let's use the most Pythonic "no loop" way to handle a monotonic stack:
    # We use a list as a stack and reduce. To handle the pop, 
    # we can use a helper function that is called by reduce.
    # But the pop is a loop. 
    # Is there a way to pop without a loop?
    # We can use `itertools.dropwhile`.
    
    from itertools import dropwhile
    
    # For each j, we want to find L[j]: the nearest index to the left where H[L[j]] > H[j].
    # We maintain a stack of indices whose heights are decreasing.
    # For H[j], we drop elements from the stack that are smaller than H[j].
    
    # Since we can't mutate the stack in a way that requires a loop, 
    # we use reduce to pass the stack along.
    
    # L[j] = stack[0] after dropping elements < H[j]
    # new_stack = [j] + list(dropwhile(lambda x: H[x] < H[j], stack))
    
    # This is still O(N) per element if we are not careful, 
    # but dropwhile is an iterator. 
    # The total number of elements dropped across all j is N.
    # So this is O(N).
    
    # Let's implement this.
    
    # H is 0-indexed internally.
    # For j = 0 to N-1:
    #   (stack, L_vals) = reduce(...)
    #   L[j] = stack[0] if stack else -1
    #   stack = [j] + list(dropwhile(lambda x: H[x] < H[j], stack))
    
    # Then we use the L[j] to update a difference array.
    # diff[L[j]+1] += 1, diff[j] -= 1
    # Then prefix sum the diff array.
    
    # To avoid loops for the final sum, we use reduce.
    
    # We need H to be accessible inside the lambda.
    # We use a list for H.
    
    # The state for reduce: (stack, L_list)
    # We use a list for L_list and append to it.
    
    # Note: list(dropwhile(...)) creates a new list. 
    # Total complexity is O(N) because each element