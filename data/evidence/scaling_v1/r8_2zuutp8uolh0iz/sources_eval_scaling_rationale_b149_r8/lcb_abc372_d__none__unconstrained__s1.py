The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. For this specific problem, which requires counting elements that maintain a "running maximum" property, the most efficient approach is to process the sequence from right to left. Since we cannot use loops, we utilize `functools.reduce` to maintain a state (the current visible buildings) and a list to accumulate results, leveraging Python's ability to handle recursion-like state transitions within a single expression.

To solve this, we observe that for a fixed $i$, we are looking for the number of $j > i$ such that $H_j > \max(H_{i+1}, \dots, H_{j-1})$. This is equivalent to counting how many elements to the right of $i$ would be visible if one looked from building $i$, provided we only count buildings that are taller than all buildings between them and $i$. However, the condition is simpler: $j$ satisfies the condition if $H_j$ is a prefix maximum of the sequence $H_{i+1}, \dots, H_N$. 

Wait, the condition is: "no building taller than Building $j$ between $i$ and $j$". This means for a fixed $i$, we are counting $j > i$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is exactly the definition of elements that would be added to a monotonic stack when iterating from $i+1$ to $N$. 

Actually, a more efficient way to think about this is: for a fixed $j$, it is counted for all $i < j$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is true for all $i$ from the index of the first building to the left of $j$ that is taller than $H_j$ (exclusive) up to $j-1$. 

Let $L_j$ be the index of the nearest building to the left of $j$ such that $H_{L_j} > H_j$. The condition is satisfied for all $i$ such that $L_j \le i < j$. The number of such $i$ is $j - L_j$. We can use a Fenwick tree or Segment tree to sum these contributions, but since we need the answer for each $i$, we can use a difference array (or similar) and prefix sums.

```python
import sys
from functools import reduce

def solve():
    # Read N and H
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find for each j, the nearest index L_j < j such that H[L_j] > H[j].
    # We use a monotonic stack to find L_j. 
    # Since loops are forbidden, we use reduce to simulate the stack process.
    
    # stack stores indices. reduce returns (current_stack, list_of_L_j)
    def find_left_greater(acc, idx):
        stack, l_values = acc
        # Remove elements from stack that are smaller than current height
        # We use a helper function to simulate the 'while' loop using recursion or 
        # a clever slice, but since we can't use loops, we use a trick with 
        # a custom function and reduce or a list comprehension.
        # However, the constraint is "no for or while loops".
        # We can use a recursive-like structure via a helper function.
        
        def pop_smaller(s):
            if not s or h[s[-1]] > h[idx]:
                return s
            return pop_smaller(s[:-1])
        
        new_stack = pop_smaller(stack)
        l_val = new_stack[-1] if new_stack else 0
        return (new_stack + [idx + 1], l_values + [l_val])

    # To avoid recursion depth issues and loops, we use a different approach for L_j.
    # We can use the fact that we need to count i in [L_j, j-1].
    # This is equivalent to: for each j, add 1 to range [L_j, j-1].
    # Then the answer for i is the sum of additions at index i.
    
    # Because we cannot use loops, we use a technique to find L_j using a 
    # functional approach. Since N=2e5, recursion will hit limits.
    # We will use a list-based approach with reduce to build the L array.
    # To simulate the while loop in the stack, we can use a trick with 
    # a helper function and a list.
    
    # Actually, the most reliable way to avoid loops/recursion for this 
    # specific problem is to realize we can't easily implement a monotonic 
    # stack without loops. Let's use the property that we can use 
    # map/filter/reduce.
    
    # Let's redefine: for each i, we want to count j > i such that 
    # max(H_{i+1} ... H_{j-1}) < H_j.
    # This is equivalent to counting j > i such that H_j is a 
    # "right-side" visible building.
    
    # Correct approach: For each j, it contributes to i in range [L_j, j-1].
    # L_j is the index of the first building to the left taller than H_j.
    # We can find L_j for all j using a stack. To do this without loops:
    # We use a recursive function with @lru_cache or similar, but that's risky.
    # Instead, we use the 'bisect' module on a sorted list of indices 
    # of buildings taller than current, but that's for different problems.
    
    # Let's use the property: j satisfies the condition for i if 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means for a fixed i, we are counting the number of 
    # elements in the sequence H_{i+1}...H_N that are strictly 
    # greater than all preceding elements in that subsequence.
    
    # This is exactly the number of elements that would remain in a 
    # monotonic increasing stack if we processed H_{i+1}...H_N.
    # But we need this for all i.
    
    # Let's use the L_j logic. To find L_j without loops:
    # We can use a divide and conquer approach implemented via 
    # a recursive function (which is allowed as long as it's not a loop).
    # But the constraint says "no for or while loops". 
    # Recursion is allowed.
    
    sys.setrecursionlimit(300000)
    
    def get_l_values(arr):
        n = len(arr)
        l_vals = [0] * n
        def compute(stack, idx):
            if idx == n:
                return
            # Simulate while stack and h[stack[-1]] < h[idx]: stack.pop()
            # Since we can't use while, we use a helper
            def pop_stack(s):
                if not s or h[s[-1]-1] > h[idx]:
                    return s
                return pop_stack(s[:-1])
            
            s_after_pop = pop_stack(stack)
            l_vals[idx] = s_after_pop[-1] if s_after_pop else 0
            compute(s_after_pop + [idx + 1], idx + 1)
            
        compute([], 0)
        return l_vals

    # The above recursion is still O(N^2) in worst case due to s[:-1] 
    # and stack passing. Let's use a mutable list and a recursive function.
    
    l_vals = [0] * n
    stack = []
    def solve_recursive(idx):
        if idx == n:
            return
        while_sim(idx)
        solve_recursive(idx + 1)

    # Wait, I cannot use 'while' inside while_sim.
    # Let's use a different approach. For each i, the answer is 
    # the number of j > i such that H_j > max(H_{i+1}...H_{j-1}).
    # This is equivalent to: j is counted if H_j is a prefix maximum 
    # of the suffix starting at i+1.
    
    # Let's use the L_j logic with a mutable stack and a recursive 
    # function to replace the while loop.
    
    stack = []
    l_vals = [0] * n
    def pop_func(idx):
        if stack and h[stack[-1]] < h[idx]:
            stack.pop()
            return pop_func(idx)
        return None

    def iterate(idx):
        if idx == n:
            return
        pop_func(idx)
        l_vals[idx] = stack[-1] + 1 if stack else 0
        stack.append(idx)
        iterate(idx + 1)

    iterate(0)
    
    # Now we have L_j for each j (1-indexed).
    # For each j, it contributes to i in range [L_j, j-1].
    # We need to count for each i, how many j > i satisfy L_j <= i.
    # This is: for a fixed i, count j such that j > i and L_j <= i.
    # This is equivalent to: (total j > i) - (count j > i such that L_j > i).
    # L_j is the index of the first building to the left taller than H_j.
    # L_j > i means all buildings between i and j are shorter than H_j.
    # Wait, the condition is: no building taller than H_j between i and j.
    # That means max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is exactly L_j <= i < j.
    
    # To calculate this for all i:
    # Create a difference array. For each j:
    # range [L_j, j-1] gets +1.
    # diff[L_j] += 1, diff[j] -= 1.
    
    diff = [0] * (n + 1)
    # Use map to simulate the loop for the difference array
    def update_diff(j):
        # j is 0-indexed, L_j is 0-indexed (adjusted)
        # L_j is the index of the building taller than H_j.
        # The condition L_j <= i < j.
        # L_j is l_vals[j] (which is 1-indexed index of the taller building)
        # So i can be l_vals[j]-1, ..., j-1.
        # In 0-indexing: i is in [l_vals[j]-1, j-1] if we consider i as the 
        # building index. But the problem asks for i = 1...N.
        # For a fixed i (0-indexed), we count j > i such that L_j <= i.
        # L_j is the index of the first building to the left of j taller than H_j.
        # If no such building exists, L_j = -1.
        # Condition: max(H_{i+1}...H_{j-1}) < H_j.
        # This is true if the first building to the left of j taller than H_j 
        # is at index <= i.
        # So for each j, it contributes to i in range [L_j, j-1].
        # Using the l_vals calculated (which are 1-indexed indices of taller buildings):
        # L_j_0indexed = l_vals[j] - 1.
        # Range of i (0-indexed): [l_vals[j]-1, j-1].
        # Note: if l_vals[j] is 0, the range is [0, j-1].
        # However, the condition is j > i. So i is at most j-1.
        # The range of i is [max(0, l_vals[j]-1), j-1].
        # But wait, if i = L_j_0indexed, then building i is the one taller than H_j.
        # The buildings between i and j are H_{i+1}...H_{j-1}.
        # All of these are smaller than H_j by definition of L_j.
        # So i = L_j_0indexed is allowed.
        # The range of i is [max(0, l_vals[j]-1), j-1].
        # But we must exclude i = j.
        # For each j from 0 to n-1:
        # start = max(0, l_vals[j]-