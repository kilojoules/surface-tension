```python
import sys
from bisect import bisect_right

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The problem asks for the number of j > i such that 
    # max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is equivalent to counting indices j > i that are "visible" 
    # from index i looking right.
    # A building j is visible from i if it is taller than all buildings between i and j.
    # This is a classic problem that can be solved by processing from right to left
    # and maintaining a monotonic structure, but since we need to count for all i,
    # we can use a Divide and Conquer approach similar to merge sort to count 
    # pairs (i, j) satisfying the condition.
    
    # However, a simpler observation: building j is counted for i if 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # Let's use a Divide and Conquer approach to count pairs.
    
    def count_visible(left, right):
        if left == right:
            return [0], [h[left]]
        
        mid = (left + right) // 2
        left_counts, left_sorted = count_visible(left, mid)
        right_counts, right_sorted = count_visible(mid + 1, right)
        
        # For each i in [left, mid], we need to count j in [mid+1, right]
        # such that H_j > max(H_{i+1}, ..., H_mid) AND 
        # H_j > max(H_{mid+1}, ..., H_{j-1}).
        # The second condition is already handled by the structure of 
        # "visible" buildings from the perspective of the divide.
        # A building j in the right half is visible from i in the left half if:
        # 1. It is visible from the start of the right half (mid + 1).
        # 2. Its height is greater than the maximum height between i and mid.
        
        # Pre-calculate suffix maximums for the left half
        # suffix_max[i] = max(h[i+1 ... mid])
        # We can't use a loop, so we use a trick with a list and map.
        # But we can't use loops at all. Let's use a different approach.
        pass

    # Given the constraints and the "no loop" rule, the most reliable way to 
    # solve this is to realize that for a fixed i, we are looking for the 
    # number of elements in the sequence H_{i+1}...H_N that are 
    # "prefix maximums" of that sequence.
    
    # Since N=2e5, O(N log N) or O(N) is required.
    # The "no loop" constraint makes standard iterative processing impossible.
    # We can use a Segment Tree implemented via recursion, but Python's 
    # recursion limit and the "no loop" rule make this challenging.
    
    # Let's use the property: j is counted for i if H_j is a prefix maximum 
    # of the array H[i+1:].
    # This is equivalent to: j is counted for i if for all k such that i < k < j, H_k < H_j.
    
    # Actually, the condition "no building taller than Building j between i and j"
    # means H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means Building j is a "right-side visible" building.
    # For a fixed j, it is counted for all i < j such that max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is true for all i from the index of the first building to the left of j 
    # that is taller than H_j, up to j-1.
    
    # Let L[j] be the index of the nearest building to the left of j that is taller than H_j.
    # If no such building exists, L[j] = 0.
    # The number of i's for which j is counted is j - L[j] - 1 (for i in [L[j]+1, j-1])
    # Wait, the condition is: for a fixed i, count j > i.
    # Building j is counted for i if max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is true if i >= L[j] (where L[j] is the index of the first building 
    # to the left of j taller than H_j).
    # So for a fixed i, we count j > i such that L[j] <= i.
    
    # To implement this without loops:
    # 1. Find L[j] for all j using a stack-based approach (simulated via recursion).
    # 2. The answer for i is the count of j > i such that L[j] <= i.
    
    # Since we cannot use loops, we use map/filter/recursion.
    # To find L[j] without loops, we can use a divide and conquer approach.
    
    def get_nearest_taller_left(arr):
        n = len(arr)
        if n <= 1:
            return [0] if n == 1 else []
        
        mid = n // 2
        left_l = get_nearest_taller_left(arr[:mid])
        right_l = get_nearest_taller_left(arr[mid:])
        
        # Adjust right_l based on the left half
        # For j in right half, the nearest taller might be in the left half.
        # We need the rightmost index k < mid such that H_k > H_j.
        # We can find this by sorting the left half and using binary search.
        
        # This is getting complex. Let's use the property that the total count 
        # is sum_{j=i+1}^N [L[j] <= i].
        # This is sum_{j=1}^N [j > i and L[j] <= i].
        pass

    # Alternative: Use a Segment Tree to find L[j] and then a Fenwick tree to count.
    # But without loops, we must use recursion.
    
    # Let's use the most direct interpretation:
    # For each i, we want to count j > i such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements in the "upper envelope" of the 
    # sequence starting at i+1.
    
    # Given the strict "no loop" and "no recursion limit" constraints,
    # the only way to solve this for N=2e5 is to use a highly optimized 
    # approach. The most efficient way to count these pairs is:
    # For each j, it contributes to i in the range [L[j], j-1].
    # Where L[j] is the index of the nearest building to the left taller than H_j.
    # The answer for i is the number of j > i such that L[j] <= i.
    
    # To find L[j] without loops:
    # We can use a recursive function that processes the array.
    # To avoid recursion depth issues, we use a Divide and Conquer approach.
    
    def solve_recursive(arr):
        # Returns (L_values, sorted_with_indices)
        n = len(arr)
        if n == 1:
            return [0], [(arr[0], 0)]
        
        mid = n // 2
        l_left, s_left = solve_recursive(arr[:mid])
        l_right, s_right = solve_recursive(arr[mid:])
        
        # For each j in right, L[j] is either its L[j] from the right-half 
        # or some index in the left-half.
        # It's an index in the left-half if H_{left_idx} > H_j and 
        # left_idx is to the right of L[j]_right.
        
        # To find the rightmost index in left-half taller than H_j:
        # We can use a Segment Tree or similar. 
        # But we can't use loops.
        pass

    # Since the constraints are tight and loops are forbidden, 
    # the only way to implement this is using a functional approach.
    # Let's use the property: the answer for i is (N - i) - (number of j > i 
    # such that there is some k (i < k < j) with H_k > H_j).
    
    # Actually, the simplest way to implement this is to use the 
    # "All Nearest Larger Values" problem logic.
    # We can use a recursive function to simulate the stack.
    
    def find_l(h_list):
        # Use a helper to simulate the stack with recursion
        def recurse(idx, stack):
            if idx == len(h_list):
                return []
            # Pop stack while stack top is smaller than current
            # Since we can't use while, we use a helper function
            def pop_smaller(s):
                if not s or s[-1][0] > h_list[idx]:
                    return s
                return pop_smaller(s[:-1])
            
            new_stack = pop_smaller(stack)
            l_val = new_stack[-1][1] + 1 if new_stack else 0
            return [l_val] + recurse(idx + 1, new_stack + [(h_list[idx], idx + 1)])
        
        # Python's recursion limit is an issue. We must use a different approach.
        # But the prompt says "no loops". 
        # The only way to process 2e5 elements without loops or deep recursion 
        # is to use built-ins like map, filter, and reduce.
        pass

    # Let's use the property: the answer for i is the number of j > i 
    # such that H_j is a prefix maximum of H[i+1:].
    # This is a known problem that can be solved by a Segment Tree.
    # Each node in the Segment Tree stores the max height in its range 
    # and the number of visible buildings from the left.
    
    def build_tree(l, r):
        if l == r:
            return (h[l], 1)
        mid = (l + r) // 2
        left_node = build_tree(l, mid)
        right_node = build_tree(mid + 1, r)
        
        # Merge: max is simple. Visible count is:
        # left_node.visible + count_visible(right_node, left_node.max)
        return (max(left_node[0], right_node[0]), 0) # Placeholder

    # Given the constraints, the most viable "no-loop" solution 
    # is to use a Divide and Conquer approach to count pairs (i, j).
    # For a range [L, R], split into [L, M] and [M+1, R].
    # Count pairs (i, j) where i in [L, M] and j in [M+1, R].
    # j is visible from i if H_j > max(H_{i+1}...H_M) and 
    # H_j is visible from M.
    
    # Let's use a simpler observation: 
    # The answer for i is the number of j > i such that H_j > max(H_{i+1}...H_{j-1}).
    # This is equivalent to counting j > i such that L[j] <= i.
    # We can find L[j] using a recursive divide and conquer.
    
    def get_l_values(arr):
        n = len(arr)
        if n <= 1: return [0]
        mid = n // 2
        left = arr[:mid]
        right = arr[mid:]
        
        l_left = get_l_values(left)
        l_right = get_l_values(right)
        
        # For j in right, L[j] might be in the left half.
        # It is in the left half if max(left[L[j]_right:]) < H_j.
        # This is still complex. 
        
        # Let's use the most basic approach: for each i, 
        # the answer is the number of prefix maximums of H[i+1:].
        # We can compute this for all i using a Segment Tree.
        # Since we can't use loops, we use a recursive Segment