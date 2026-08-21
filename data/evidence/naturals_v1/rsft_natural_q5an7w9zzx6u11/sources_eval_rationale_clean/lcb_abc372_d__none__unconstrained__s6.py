The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. In this specific problem, we need to count indices $j > i$ such that $\max(H_{i+1}, \dots, H_{j-1}) < H_j$. This is equivalent to finding how many elements to the right of $i$ are "visible" if we look from building $i$, but the condition is slightly different: it asks for $j$ such that no building between $i$ and $j$ is taller than $H_j$. This means $H_j$ must be a new maximum for the range $(i, j]$.

To solve this without loops, I will use a recursive approach with a Segment Tree (implemented via mapping/reduction) or a Divide and Conquer strategy. However, since Python's recursion limit and the lack of loops make standard Segment Trees difficult, the most idiomatic "loop-free" way to handle this range-based counting is using a Divide and Conquer algorithm implemented via recursion, utilizing list slicing and `map` for processing.

```python
import sys

# Increase recursion depth for deep Divide and Conquer trees
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition: j > i satisfies the condition if H[j] > max(H[i+1...j-1])
    # This is equivalent to saying H[j] is a prefix maximum of the sequence H[i+1...N]
    # However, the condition is simply: for a fixed i, we count j > i such that
    # H[j] is greater than all elements between i and j.
    # This means we are counting the number of elements in the sequence H[i+1...N]
    # that are strictly greater than all preceding elements in that subsequence.
    
    # Let f(i) be the number of j > i satisfying the condition.
    # For a fixed i, we are looking at the sequence S = [H[i+1], H[i+2], ..., H[N]].
    # We want to count how many H[j] are greater than max(H[i+1]...H[j-1]).
    # This is exactly the count of prefix maximums of the sequence H[i+1...N].
    
    # To solve this for all i efficiently without loops:
    # We can use a Divide and Conquer approach.
    # For a range [L, R], we calculate contributions across the midpoint.
    
    def get_prefix_max_counts(arr):
        # Returns a list where result[i] is the number of prefix maximums in arr[i:]
        # This is still hard without loops. 
        # Let's redefine: we need to count j > i such that H[j] > max(H[i+1...j-1]).
        # This is equivalent to: H[j] is a prefix maximum of the suffix starting at i+1.
        pass

    # Given the constraints and the "no loop" rule, the most effective way to 
    # implement this is to realize that for a fixed i, the answer is the number 
    # of elements in the suffix H[i+1:] that are larger than all elements 
    # appearing before them in that suffix.
    
    # This is a classic problem solvable by a Segment Tree or Fenwick Tree, 
    # but those require loops. A Divide and Conquer approach can work.
    
    def count_visible(left, right):
        if left == right:
            return [0]
        
        mid = (left + right) // 2
        # Divide
        left_res = count_visible(left, mid)
        right_res = count_visible(mid + 1, right)
        
        # Combine: For i in [left, mid], we need to count j in [mid+1, right]
        # such that H[j] > max(H[i+1...mid]) AND H[j] is a prefix max of H[mid+1...right]
        # relative to the max of the left bridge.
        
        # This is getting complex for a loop-free constraint. 
        # Let's use the property: the answer for i is the number of prefix maximums 
        # of the sequence H[i+1...N].
        
        # Let's use a recursive function with memoization or a functional 
        # approach to simulate the process.
        pass

    # Alternative: The number of prefix maximums of H[i+1...N] can be found by
    # building a Cartesian tree. In a Cartesian tree (max-heap), the number of 
    # prefix maximums of a suffix is related to the path to the root.
    
    # Since I cannot use loops, I will use a recursive function to build the 
    # answer array by processing the range and using the fact that 
    # the maximum element in a range splits the problem.
    
    def compute(l, r):
        if l >= r:
            return [0] * (r - l + 1)
        
        # Find index of max element in H[l:r+1]
        # Using max() and index() is allowed as they are built-ins
        sub = H[l:r+1]
        max_val = max(sub)
        m = l + sub.index(max_val)
        
        # For i < m: Building m is always a prefix maximum of H[i+1...N]
        # because it's the maximum of the whole range.
        # Any j > m that was a prefix maximum of H[m+1...N] is still one for H[i+1...N]
        # provided H[j] > H[m], but H[m] is the global max, so no j > m can be.
        # Wait, the condition is H[j] > max(H[i+1...j-1]).
        # If j = m, it's always true if H[m] is the max of H[i+1...m].
        # If j > m, H[m] is between i and j, and H[m] > H[j], so j cannot satisfy it.
        
        # So for i < m:
        # The j's that satisfy the condition are:
        # 1. The j's in [i+1, m-1] that satisfy the condition (handled by recursion)
        # 2. j = m (always satisfies because H[m] is max of H[l...r])
        # 3. No j > m can satisfy it because H[m] is in the way.
        
        # For i = m:
        # The j's are those in [m+1, r] that satisfy the condition.
        
        # For i > m:
        # The j's are those in [i+1, r] that satisfy the condition.
        
        # This structure allows a recursive calculation.
        # Let's use a helper to get the results.
        pass

    # To avoid loops and recursion depth issues, and since we need to return a list,
    # we can use a divide and conquer approach with a helper function.
    
    def solve_recursive(l, r):
        if l == r:
            return [0]
        
        # Find index of max in range [l, r]
        # We use a slice and index()
        segment = H[l:r+1]
        m = l + segment.index(max(segment))
        
        # Right side: i in [m, r]
        # For i = m, the answer is the number of prefix maximums of H[m+1...r]
        # For i > m, it's the same as the subproblem solve_recursive(m+1, r)
        
        # To find number of prefix maximums of H[m+1...r]:
        # That is exactly the answer for i = m.
        
        # Let's use a simpler observation:
        # The answer for i is: 1 + (answer for i' where i' is the index of the 
        # first element to the right of i that is taller than H[i], 
        # but only if we consider the range).
        # Actually, the answer for i is simply the number of elements 
        # in the "monotonic chain" starting from the first element to the right.
        
        # Let's use the property: 
        # ans[i] = 1 + ans[next_taller[i]] if next_taller[i] exists, else 0.
        # Wait, that's for a different problem.
        # The correct one: ans[i] = 1 + ans[index of max in H[i+1...N]] ? No.
        
        # Correct logic:
        # For a fixed i, we are counting j > i such that H[j] > max(H[i+1...j-1]).
        # This is exactly the number of prefix maximums of the sequence H[i+1...N].
        # Let f(i) be the number of prefix maximums of H[i...N].
        # f(i) = 1 + f(index of the first element in H[i+1...N] that is > H[i])
        # If no such element exists, f(i) = 1.
        # The answer for building i is f(i+1).
        
        pass

    # Since we can't use loops, we can use a recursive function with @lru_cache
    # to implement the logic: f(i) = 1 + f(next_taller_index(i))
    
    from functools import lru_cache
    
    # To find next_taller_index without loops, we can use a Segment Tree 
    # or just use the recursive property.
    # But we need next_taller_index for all i.
    # We can find all next_taller_indices using a Divide and Conquer approach.
    
    # Actually, the simplest way to implement this without loops 
    # is to use the recursive definition of the answer:
    # Let g(i) be the number of prefix maximums of H[i...N-1].
    # g(i) = 1 + g(next_taller[i]) if next_taller[i] < N else 1.
    # The answer for building i (0-indexed) is g(i+1).
    
    # To find next_taller without loops:
    # We can use a recursive function that finds the first index j > i with H[j] > H[i].
    # To make it O(N log N) or O(N), we use a Segment Tree (implemented recursively).
    
    def build_tree(l, r):
        if l == r:
            return H[l]
        mid = (l + r) // 2
        return (build_tree(l, mid), build_tree(mid + 1, r))

    # This is getting too complex for the constraints. 
    # Let's use the most direct recursive approach:
    # The number of prefix maximums of H[i...N-1] is:
    # 1 + (number of prefix maximums of H[k+1...N-1] where k is the index of 
    # the first element > H[i] in H[i+1...N-1]).
    # If no such k exists, it's just 1.
    
    # To find k efficiently: use a Segment Tree to find the first index j > i such that H[j] > H[i].
    
    # Given the "no loop" constraint, I will use a recursive function 
    # to simulate the process and `sys.setrecursionlimit`.
    
    @lru_cache(None)
    def get_ans(i):
        if i >= N:
            return 0
        # Find first j > i such that H[j] > H[i]
        # Since we can't loop, we use a helper function to find the index
        # We can use a Segment Tree implemented as a nested list/tuple
        # But building it without loops requires recursion.
        pass

    # Let's use a different approach: 
    # The answer for i is the number of elements in the sequence:
    # j1 = i+1, j2 = first index > j1 such that H[j2] > H[j1], ...
    # This is exactly the number of nodes in the path to the root in a 
    # Cartesian tree if we consider the range [i+1, N].
    
    # Final attempt strategy: 
    # Use a recursive function to find the answer for all i.
    # For a range [l, r], find the index m of the maximum element.
    # For all i in [l, m-1], the building m is a prefix maximum.
    # The number of prefix maximums in [i+1