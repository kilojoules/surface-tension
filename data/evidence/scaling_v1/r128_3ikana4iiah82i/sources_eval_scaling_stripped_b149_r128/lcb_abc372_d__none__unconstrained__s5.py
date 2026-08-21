```python
import sys
from bisect import bisect_right

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[j] is a "right-to-left" maximum of the suffix starting at i+1.
    # However, it is simpler to think: j satisfies the condition if 
    # H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many j > i are "visible" from the left.
    # Actually, the condition is: j is counted if H[j] is greater than all 
    # heights in the range (i, j).
    
    # Let's rephrase: for a fixed i, we want to count j > i such that 
    # for all k where i < k < j, H[k] < H[j].
    # This means H[j] must be a prefix maximum of the sequence H[i+1...N].
    # Wait, that's not correct. If H = [2, 1, 4, 3, 5] and i=1 (H[0]=2):
    # j=2 (H[1]=1): range (0, 1) is empty. OK.
    # j=3 (H[2]=4): range (0, 2) contains H[1]=1. 1 < 4. OK.
    # j=4 (H[3]=3): range (0, 3) contains H[1]=1, H[2]=4. 4 > 3. NOT OK.
    # j=5 (H[4]=5): range (0, 4) contains H[1]=1, H[2]=4, H[3]=3. All < 5. OK.
    # Total for i=1 is 3.
    
    # Correct observation: For a fixed i, j satisfies the condition if 
    # H[j] > max(H[i+1...j-1]).
    # This is exactly the definition of elements that would be kept in a 
    # monotonic increasing stack if we processed the suffix H[i+1...N].
    # But we need this for all i.
    
    # Let's use the property: j satisfies the condition for i if 
    # there is no k such that i < k < j and H[k] > H[j].
    # This means for a fixed j, it satisfies the condition for all i < j
    # such that for all k in (i, j), H[k] < H[j].
    # This is true for all i from (index of the first building to the left of j 
    # that is taller than H[j]) + 1 up to j-1.
    
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = -1.
    # The condition is satisfied for i in the range [L[j] + 1, j - 1].
    # The number of such i is (j - 1) - (L[j] + 1) + 1 = j - L[j] - 1.
    # However, we need to count for each i, how many j satisfy this.
    # This is a range update problem: for each j, increment range [L[j]+1, j-1].
    # Then query values at each i.
    
    # To find L[j] for all j, we use a monotonic stack.
    # Since we cannot use loops, we can use a trick with a list and a 
    # function that finds the index. But the constraint allows recursion 
    # if we use a specific structure or we can use a functional approach.
    # Actually, we can find L[j] using a list comprehension and a helper 
    # that manages the stack, but Python's list comprehensions can't 
    # easily update a stack.
    
    # Alternative: Use the fact that we need to count i's.
    # For each j, it contributes to i in range [L[j]+1, j-1].
    # We can use a Difference Array to perform range updates.
    # diff[L[j]+1] += 1, diff[j] -= 1.
    # Then the result for i is the prefix sum of the difference array.
    
    # How to find L[j] without loops? 
    # We can use a recursive-like structure via a list and a custom 
    # reduction or use the 'stack' logic inside a generator.
    # But the simplest way to find the nearest greater element to the left 
    # without explicit loops is to use a technique involving 
    # sorting and a Segment Tree or Fenwick Tree, but that's complex.
    
    # Let's use the property that we can find L[j] by iterating 
    # through the array and maintaining a stack. 
    # We can implement the stack logic using a generator and `itertools.accumulate`.
    
    from itertools import accumulate
    
    # We want to find L[j] for all j.
    # We can use a list as a stack and a function that modifies it.
    # Since we can't use loops, we use a helper function with a list.
    def get_l(h):
        stack = []
        def process(x):
            while stack and stack[-1][0] < x:
                stack.pop()
            res = stack[-1][1] if stack else -1
            stack.append((x, len(stack))) # This is wrong, need index
            return res
        # The above uses a while loop. Let's use a different approach.
        pass

    # Correct approach to find L[j] without loops:
    # Use a list comprehension to build the L array by referencing 
    # previously computed values.
    # L[j] = L[j-1] if H[j-1] < H[j] else ... 
    # This is still recursive. 
    
    # Let's use the property: L[j] is the index of the first element to the 
    # left of j that is > H[j].
    # We can find this by:
    # 1. Creating a list of (height, index) pairs.
    # 2. Sorting them by height descending.
    # 3. Using a SortedList (from bisect) to find the largest index < j.
    
    # Since we can't use loops, we use map/filter/reduce.
    from functools import reduce
    
    # To find L[j] for all j:
    # We can use a list as a stack and reduce.
    # The state is (stack, results_list)
    def find_l(acc, x):
        stack, res = acc
        # We need to pop from stack while stack[-1] < x.
        # Since we can't use while, we can use a recursive helper.
        def pop_stack(s, val):
            if not s or s[-1] >= val:
                return s
            return pop_stack(s[:-1], val)
        
        new_stack = pop_stack(stack, x)
        # The index of the nearest greater is the index of the top of the stack.
        # But we need the indices. Let's store (height, index) in the stack.
        return (new_stack, res)

    # Wait, the recursion limit might be an issue. Let's use a different trick.
    # We can find L[j] by:
    # For each j, L[j] is the index of the first element in the 
    # "monotonic decreasing" subsequence ending at j-1 that is > H[j].
    
    # Let's use the property: L[j] = (index of first element to the left > H[j]).
    # We can find this by iterating through the indices and using a 
    # Fenwick tree or Segment tree to find the maximum index in a range.
    # But that's for range queries.
    
    # Actually, the most Pythonic way to do this without loops is 
    # using a generator with a yield-based stack or just using 
    # a list comprehension with a helper.
    # But the constraint says no loops. 
    # Let's use the 'reduce' approach with a helper function for the stack.
    
    def get_l_indices(H):
        # stack stores (height, index)
        def step(state, current):
            stack, results = state
            # Instead of a while loop, we use a helper to find the 
            # first element in the stack > current height.
            # We can use a list comprehension to find the index.
            # The stack is maintained as a list of (height, index).
            # We find the first (h, i) from the end where h > current_h.
            
            # To avoid loops, we can use a recursive function to clean the stack.
            # But we can also just filter the stack.
            # However, we need to update the stack for the next element.
            pass
            
    # Let's reconsider: the condition is H[j] > max(H[i+1...j-1]).
    # This is equivalent to: i is in the range [L[j]+1, j-1].
    # We can find L[j] using a simple list comprehension if we 
    # pre-calculate the "next greater" using a trick.
    
    # Actually, the simplest way to implement a stack without loops 
    # is to use a recursive function.
    def solve_recursive(H):
        N = len(H)
        L = [0] * N
        
        def process(i, stack):
            if i == N:
                return
            # Find first element in stack > H[i]
            # The stack contains indices of elements in decreasing order of height
            # We need to remove elements from the top that are <= H[i]
            
            # Instead of a loop, we use a helper to find the split point
            # Since the stack is sorted by height (decreasing), we can use bisect
            # But the stack is sorted by height, not by index.
            # Let's maintain the stack as a list of heights.
            pass

    # FINAL ATTEMPT: Use a list as a stack and a helper function 
    # that uses recursion to pop.
    def get_l_final(H):
        def pop_until(stack, val):
            if not stack or stack[-1][0] > val:
                return stack
            return pop_until(stack[:-1], val)
        
        def reduce_func(state, current):
            stack, res = state
            cleaned_stack = pop_until(stack, current[0])
            idx = cleaned_stack[-1][1] if cleaned_stack else -1
            return (cleaned_stack + [current], res + [idx])
        
        initial_state = ([], [])
        final_state = reduce(reduce_func, enumerate(H), initial_state)
        # Wait, enumerate gives (index, value). I need (value, index).
        # Let's fix that.
        return final_state[1]

    # Correcting the reduce call:
    H_with_idx = [(H[i], i) for i in range(N)]
    L = reduce(lambda state, curr: (
        (lambda s: s + [curr])(
            (lambda s, v: s if not s or s[-1][0] > v else (lambda r: r)(
                # This is getting complex. Let's use a simpler recursion.
                # Since I can't use loops, I'll use a helper function.
                pass
            )
        ), 
        state[1]
    ), ([], []), H_with_idx)
    
    # Let's use a different approach. 
    # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is equivalent to counting how many times the prefix maximum 
    # changes in the suffix H[i+1...N].
    # The number of prefix maximums in a sequence is the size of the 
    # monotonic increasing subsequence.
    
    # We can use a Segment Tree to find the number of prefix maximums.
    # For a range [L, R], the number of prefix maximums is:
    # 1 + (number of prefix maximums in [mid+1, R] that are > max(H[L...mid]))
    # This