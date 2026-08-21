The constraint to avoid explicit loops forces a shift from imperative style to a functional paradigm, emphasizing data transformations over state mutation. To solve this problem, I will use `itertools.accumulate` to maintain a running maximum of building heights and list comprehensions to process the data. Specifically, for each building $i$, we are looking for the number of $j > i$ such that $H_j$ is greater than all heights in the range $(i, j)$. This is equivalent to counting how many times the prefix maximum of the sequence starting from $i+1$ increases.

```python
import sys
from itertools import accumulate

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are counting j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of times the running maximum increases
    # when iterating from i+1 to N.
    
    # To avoid loops, we use a list comprehension to iterate over each i.
    # For each i, we create a slice of the heights from i+1 onwards,
    # calculate the running maximums using accumulate,
    # and count how many of those maximums are "new" (i.e., they triggered an increase).
    
    # However, a naive slice and accumulate inside a comprehension is O(N^2).
    # Given N = 2*10^5, we need a more efficient approach.
    # The condition is: j satisfies the condition if H_j is a prefix maximum 
    # of the sequence [H_{i+1}, ..., H_N].
    
    # Wait, the constraint to avoid loops forces a functional style, but O(N^2) 
    # will TLE. Let's reconsider the logic.
    # For a fixed j, it satisfies the condition for i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0.
    # Then j satisfies the condition for all i such that L[j] <= i < j.
    # The number of such i is j - L[j].
    
    # To find L[j] without loops, we can use a recursive-like structure or 
    # a clever use of existing tools. Since we can't use loops, 
    # we can use a stack-based approach implemented via a reduction or 
    # a custom function called via map/comprehension.
    # Actually, the most idiomatic "no-loop" way to implement a stack 
    # is using a recursive-like process, but Python's recursion limit is an issue.
    # But wait, the prompt says "avoid explicit loops", 
    # which usually allows for high-order functions and comprehensions.
    
    # Let's use the property: j satisfies the condition for i if 
    # i is in the range [L[j], j-1].
    # We want to find for each i, the count of j > i such that L[j] <= i.
    # This is equivalent to counting j > i such that L[j] <= i.
    
    # To find L[j] efficiently without loops:
    # We can use a Divide and Conquer approach implemented via recursion.
    
    def get_left_bounds(heights):
        # Returns a list L where L[j] is the index of the nearest 
        # building to the left taller than H[j].
        # Using a stack-based approach is standard, but requires a loop.
        # To avoid loops, we can use a recursive function.
        # However, for N=2e5, we need to increase recursion depth.
        
        # Since we must avoid loops, we can use a trick with 
        # a custom class or a closure to maintain state inside a 
        # list comprehension or map.
        
        class StackState:
            def __init__(self):
                self.stack = [] # stores indices
                self.results = []

            def process(self, val):
                # This is called for each (index, height)
                # We need to simulate the stack popping
                # Since we can't loop, we use a helper function
                return self.pop_and_calc(val)

            def pop_and_calc(self, item):
                # We still need to pop elements. 
                # The only way to "loop" without 'for' or 'while' 
                # is recursion or high-order functions.
                pass
        
        # Actually, the most reliable way to solve this in O(N) 
        # without 'for/while' is to use a recursive function 
        # for the stack logic and sys.setrecursionlimit.
        
        import sys
        sys.setrecursionlimit(300000)
        
        def compute_L(idx, stack):
            if idx == len(heights):
                return []
            
            # Simulate stack popping
            def pop_stack(s):
                if not s or heights[s[-1]] > heights[idx]:
                    return s
                return pop_stack(s[:-1])
            
            # This is still potentially O(N^2) because of s[:-1]
            # Let's use a different approach.
            pass

    # Let's use the observation: 
    # The answer for i is the number of j > i such that 
    # H_j is a prefix maximum of H[i+1:].
    # This is a classic problem that can be solved by a Segment Tree 
    # or by observing that we are counting elements in a 
    # "monotonic chain" starting from i.
    
    # Given the strict "no loop" constraint and the need for O(N log N) or O(N),
    # I will use a recursive Divide and Conquer approach.
    
    import sys
    sys.setrecursionlimit(300000)

    def solve_recursive(l, r, heights):
        if l == r:
            return [0]
        
        mid = (l + r) // 2
        left_res = solve_recursive(l, mid, heights)
        right_res = solve_recursive(mid + 1, r, heights)
        
        # Count pairs (i, j) where i <= mid < j
        # j satisfies the condition if H_j > max(H_{i+1} ... H_{mid})
        # Let M_i = max(H_{i+1} ... H_{mid})
        # We need to count j > mid such that H_j > M_i and 
        # H_j is a prefix maximum of H[mid+1 ... j].
        
        # 1. Find prefix maximums of the right side
        # 2. For each i, binary search for the number of prefix maximums > M_i
        
        # To keep it "loop-free", we use list comprehensions and map.
        return left_res + right_res # This is a placeholder for the logic

    # REALIZATION: The "no loop" constraint is likely a challenge to use 
    # functional programming. But for this specific problem, 
    # the most efficient way is a stack. 
    # I can implement a stack using a recursive function.
    
    import sys
    sys.setrecursionlimit(300000)

    def get_l_bounds(h):
        # l_bounds[j] = index of nearest building to the left taller than h[j]
        # We use a recursive function to simulate the stack
        def recurse(idx, stack, results):
            if idx == len(h):
                return results
            
            def pop_until(s):
                if not s or h[s[-1]] > h[idx]:
                    return s
                # Using s.pop() is a mutation, but the prompt 
                # forbids 'for/while' loops, not mutation.
                s.pop()
                return pop_until(s)
            
            current_stack = pop_until(stack)
            results.append(current_stack[-1] + 1 if current_stack else 0)
            current_stack.append(idx)
            return recurse(idx + 1, current_stack, results)
        
        return recurse(0, [], [])

    # After getting L[j], the answer for i is:
    # count j > i such that L[j] <= i.
    # This can be solved by:
    # 1. Creating events: at index L[j], a "valid j" starts; at index j, it ends.
    # 2. Or more simply: the answer for i is (number of j > i with L[j] <= i).
    # This is: (total j with L[j] <= i) - (number of j <= i with L[j] <= i).
    # Since L[j] < j is always true, the second term is just the count of j <= i.
    # So: ans[i] = (count j in [0, N-1] such that L[j] <= i) - (i + 1).
    # Note: we must handle the 0-indexing and the fact that j > i.
    
    # Let's refine:
    # For a fixed i, we want count of j in {i+1, ..., N-1} such that L[j] <= i.
    # Let f(i) = count of j in {0, ..., N-1} such that L[j] <= i.
    # The answer for i is f(i) - (count of j in {0, ..., i} such that L[j] <= i).
    # Since L[j] <= j is always true, the second term is simply (i + 1).
    # So ans[i] = f(i) - (i + 1).
    # f(i) can be computed by sorting L and using binary search.

    # Implementation:
    h_vals = h
    l_bounds = get_l_bounds(h_vals)
    
    # To compute f(i) without loops:
    # sorted_l = sorted(l_bounds)
    # f(i) = bisect_right(sorted_l, i)
    
    import bisect
    sorted_l = sorted(l_bounds)
    
    # Use map to calculate ans for each i from 0 to N-1
    # Note: the problem asks for i = 1 to N.
    # For i-th building (0-indexed), we look at j > i.
    # The condition is L[j] <= i.
    # But wait, the condition is "no building taller than Building j BETWEEN i and j".
    # This means max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is exactly what L[j] <= i means if L[j] is the index of the 
    # first building to the left of j that is TALLER than H_j.
    # If H_j is the tallest in [i+1, j], then the first building 
    # taller than H_j must be at some index <= i.
    
    # Example: H = [2, 1, 4, 3, 5]
    # j=0: H=2, L=0 (none)
    # j=1: H=1, L=1 (H[0]=2 > 1)
    # j=2: H=4, L=0 (none)
    # j=3: H=3, L=3 (H[2]=4 > 3)
    # j=4: H=5, L=0 (none)
    # L = [0, 1, 0, 3, 0]
    # i=0: j in {1,2,3,4}. L[j] <= 0? j=2(0<=0), j=4(0<=0). Also j=1(1<=0 False), j=3(3<=0 False).
    # Wait, the sample says for i=1, j=2,3,5 are valid.
    # My L logic: j=1(H=1), L[1]=0. 0 <= 0 is True. So j=1 is valid.
    # j=2(H=4), L[2]=0. 0 <= 0 is True. So j=2 is valid.
    # j=3(H=3), L[3]=2. 2 <= 0 is False.
    # j=4(H=5), L[4