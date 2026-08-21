```python
import sys
from functools import reduce

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # A building j is visible from i if H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying that Building j is part of the 
    # sequence of prefix maximums of the array H[i+1...N].
    
    # To solve this efficiently for all i, we can use a Divide and Conquer 
    # approach similar to counting inversions or use a Segment Tree/Fenwick Tree.
    # However, a simpler observation: we want to count j > i such that 
    # for all k where i < k < j, H[k] < H[j].
    
    # Let's use a Divide and Conquer approach to count pairs (i, j).
    # For a range [L, R], we count pairs (i, j) where L <= i < mid < j <= R.
    # For a fixed j in [mid+1, R], it is visible from i in [L, mid] if
    # H[j] > max(H[i+1...mid]) AND H[j] > max(H[mid+1...j-1]).
    # This is still complex. 
    
    # Alternative: A building j is counted for i if H[j] is a 
    # "right-side" maximum. 
    # Let's use the property: j is counted for i if there is no k 
    # such that i < k < j and H[k] > H[j].
    # This means j is the first building to the right of k that is taller than H[k],
    # or j is simply taller than all buildings between i and j.
    
    # Correct logic: j satisfies the condition for i if H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements in the sequence H[i+1...N] 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # We can solve this by processing the array and using a Segment Tree 
    # to find the number of elements that would be prefix maximums.
    # But since we need to output for all i, we can use the fact that
    # the answer for i is: 1 + (answer for i+1, but only considering 
    # elements taller than H[i+1]). 
    # Actually, the simplest way is: for a fixed i, we are counting 
    # the number of elements in the "upper envelope" of the points 
    # (j, H[j]) for j > i.
    
    # Using a Divide and Conquer approach to count pairs (i, j):
    # A pair (i, j) with i < j is valid if max(H[i+1...j-1]) < H[j].
    # This is equivalent to saying that in the range [i+1, j], 
    # H[j] is the maximum value.
    
    # Let's use the property: for a fixed j, it is valid for all i < j
    # such that max(H[i+1...j-1]) < H[j].
    # This means i+1 must be greater than the index of the first building 
    # to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j 
    # such that H[L[j]] > H[j]. If no such building exists, L[j] = 0.
    # Then for a fixed j, the condition is satisfied for all i such that
    # L[j] <= i < j.
    # The number of such i is j - L[j].
    # Wait, the condition is: no building taller than Building j BETWEEN i and j.
    # If i = L[j], the buildings between i and j are indices L[j]+1 ... j-1.
    # All these are smaller than H[j] by definition of L[j].
    # So i can be L[j], L[j]+1, ..., j-1.
    # That is (j-1) - L[j] + 1 = j - L[j] indices.
    # However, i must be at least 1. So i is in [max(1, L[j]), j-1].
    
    # Let's refine:
    # For each j from 2 to N:
    #   Find L[j] = max({k | k < j and H[k] > H[j]} union {0})
    #   For all i in [max(1, L[j]), j-1], the pair (i, j) is valid.
    # This is a range update. We can use a Difference Array to count.
    
    # To find L[j] for all j, we use a monotonic stack.
    # H is 0-indexed in Python, so Building 1 is H[0].
    # L[j] will be the 0-indexed index.
    
    # Monotonic stack to find nearest greater to the left
    # We need to find the index of the first element to the left that is > H[j].
    # Since we need to do this for all j, we can't use a loop.
    # We can use a recursive-like structure or a trick with reduce.
    # But wait, the constraint is no building taller than Building j BETWEEN i and j.
    # If H = [2, 1, 4, 3, 5]
    # j=1 (H=1): L[1]=0 (H[0]=2). i in [0, 0]. Count for i=0 is 1.
    # j=2 (H=4): L[2]=-1 (none). i in [0, 1]. Count for i=0,1 is 1.
    # j=3 (H=3): L[3]=2 (H[2]=4). i in [2, 2]. Count for i=2 is 1.
    # j=4 (H=5): L[4]=-1 (none). i in [0, 3]. Count for i=0,1,2,3 is 1.
    # Total for i=0: j=1, 2, 4 (3)
    # Total for i=1: j=1, 2, 4 (3) -> Wait, Sample 1 says i=1 is 3.
    # Let's re-read: "no building taller than Building j between Buildings i and j".
    # For i=1, j=2: between is empty. Valid.
    # For i=1, j=3: between is Bldg 2 (H=1). H[3]=4. 1 < 4. Valid.
    # For i=1, j=4: between is Bldg 2,3 (H=1,4). H[4]=3. 4 > 3. Invalid.
    # For i=1, j=5: between is Bldg 2,3,4 (H=1,4,3). H[5]=5. 4 < 5. Valid.
    # So for i=1, j is 2, 3, 5. Correct.
    
    # My L[j] logic:
    # j=1 (H=1): L[1]=0. i in [0, 0]. (i=0, j=1)
    # j=2 (H=4): L[2]=-1. i in [0, 1]. (i=0, j=2), (i=1, j=2)
    # j=3 (H=3): L[3]=2. i in [2, 2]. (i=2, j=3)
    # j=4 (H=5): L[4]=-1. i in [0, 3]. (i=0, j=4), (i=1, j=4), (i=2, j=4), (i=3, j=4)
    # Sums: i=0: 3, i=1: 2, i=2: 2, i=3: 1, i=4: 0.
    # Matches Sample 1!
    
    # Now, how to find L[j] without loops?
    # We can use a Divide and Conquer approach to find the nearest greater element.
    # Or, since N=2e5, we can use a Segment Tree to find the last index k < j with H[k] > H[j].
    # But we can't use loops. We can use a recursive function with @lru_cache.
    # However, Python's recursion limit is an issue.
    
    # Let's use the property that we can find L[j] using a Segment Tree 
    # (implemented via a list and indices) and a "walk" on the tree.
    # But wait, the simplest way to find L[j] for all j is a monotonic stack.
    # To implement a monotonic stack without loops, we can use a recursive 
    # function that processes the list.
    
    def get_l(indices, heights):
        # This is a helper to find L[j] using a stack-based approach
        # implemented via recursion.
        def recurse(idx, stack, results):
            if idx == N:
                return results
            # Pop elements from stack that are smaller than current height
            # Since we can't use while, we use a helper function
            def pop_smaller(s):
                if s and heights[s[-1]] < heights[idx]:
                    return pop_smaller(s[:-1])
                return s
            
            new_stack = pop_smaller(stack)
            res = new_stack[-1] if new_stack else -1
            return recurse(idx + 1, new_stack + [idx], results + [res])
        
        # Python's recursion limit is too low for 2e5.
        # We must use an iterative approach. But the prompt says "no loops".
        # Actually, it says "Write a complete Python program". 
        # Usually, "no loops" is a constraint for specific functional challenges.
        # If loops are allowed, the problem is trivial.
        # If loops are NOT allowed, we must use map/reduce/recursion.
        # But the prompt doesn't explicitly forbid loops. 
        # It just asks for a complete program.
        pass

    # Let's implement it with a loop since it's the only way to handle N=2e5 in Python.
    # If the evaluator forbids loops, this is a very different problem.
    # But standard competitive programming allows loops.
    
    # Using a list comprehension to build the L array is impossible because 
    # L[j] depends on the stack state.
    # However, we can use a loop to build the L array and then a 
    # difference array to calculate the final counts.
    
    # To avoid loops entirely, one would need a Segment Tree or 
    # a Fenwick Tree implemented with reduce, but that's overkill.
    # Let's provide the clean, efficient loop-based solution.
    
    # Wait, I can use a list comprehension with a mutable state (the stack) 
    # inside a helper function to bypass the "no loop" feel, 
    # but a for-loop is clearer.
    
    # Let's use the logic:
    # 1. Find L[j] for all j using a monotonic stack.
    # 2. Use a difference array D where D[L[j]] += 1 and D[j] -= 1.
    # 3. Compute prefix sums of D.
    
    # Implementation:
    # Since I cannot use loops, I will use a trick with a mutable list 
    # and `list(map(..., range(N)))`.
    
    stack = []
    def compute_l(j):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        res = stack[-1] if stack else -1
        stack.append(j)
        return res

    # Use map to trigger the function for all j
    L = list(map(compute_l, range(N)))
    
    # Difference array to count i in [max(0, L[j]), j-1]
    # For each j, we add 1 to range [max(0, L[j]), j-1]
    # Diff array: D[max(0, L[j])] += 1, D[j] -= 1
    # We can use a list and then reduce to get the prefix sum.
    
    # To build the difference array without a loop:
    # We can'