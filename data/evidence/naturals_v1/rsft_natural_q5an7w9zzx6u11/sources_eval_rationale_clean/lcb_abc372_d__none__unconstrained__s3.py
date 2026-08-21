The constraint to avoid explicit loops forces a shift from imperative programming to a functional paradigm, emphasizing data transformations over state mutation. In idiomatic Python, this encourages the use of high-level abstractions like `map`, `filter`, and `functools.reduce`, which can often be more concise and expressive for certain types of data processing. To solve this problem without loops, I will use `functools.reduce` to simulate a stack-based approach, processing the buildings from right to left to count visible buildings.

```python
import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is visible from Building i if it is a 
    # "right-side" prefix maximum starting from index i+1.
    # However, the problem asks for the count of j > i such that 
    # max(H_{i+1} ... H_{j-1}) < H_j.
    # This is equivalent to saying Building j is visible from Building i
    # if it is part of the monotonic increasing stack when scanning from i+1 to N.
    
    # To avoid loops, we use reduce to process the array from right to left.
    # We maintain a stack of buildings that could be visible to buildings to the left.
    # For a building i, the buildings j > i that satisfy the condition are
    # exactly the buildings that form a strictly increasing sequence of heights
    # when looking from i towards the right.
    
    # Wait, the condition is: no building between i and j is taller than H_j.
    # This means H_k < H_j for all i < k < j.
    # This is exactly the definition of buildings that would remain in a 
    # monotonic stack if we processed from j down to i.
    # Actually, for a fixed i, the valid j's are those where H_j is a 
    # new maximum encountered while scanning from i+1 to N.
    # No, that's not correct. Let's re-read: "no building taller than Building j between i and j".
    # Example 1: H = [2, 1, 4, 3, 5], i=1 (H_1=2).
    # j=2 (H_2=1): between 1 and 2 is empty. Valid.
    # j=3 (H_3=4): between 1 and 3 is H_2=1. 1 < 4. Valid.
    # j=4 (H_4=3): between 1 and 4 are H_2=1, H_3=4. 4 > 3. Invalid.
    # j=5 (H_5=5): between 1 and 5 are H_2=1, H_3=4, H_4=3. All < 5. Valid.
    # Result for i=1 is 3.
    
    # This means for a fixed i, we are looking for j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is simply the count of elements in the sequence H_{i+1}...H_N 
    # that are strictly greater than all preceding elements in that subsequence.
    
    # This is equivalent to: for a fixed i, how many j > i are "visible" 
    # if we look right. But the condition is about H_j being the tallest 
    # among the buildings between i and j.
    # This is exactly the number of elements that would be pushed onto a 
    # monotonic increasing stack when iterating from i+1 to N.
    
    # However, the problem is symmetric to: for a fixed j, how many i < j 
    # satisfy the condition? 
    # Building j is visible from i if H_k < H_j for all i < k < j.
    # This means i must be greater than the index of the first building to the 
    # left of j that is taller than H_j.
    # Let L_j be the index of the nearest building to the left of j such that H_{L_j} > H_j.
    # If no such building exists, L_j = 0.
    # Then for a fixed j, the valid i's are i = L_j, L_j+1, ..., j-1.
    # Wait, the condition is "between i and j". 
    # If i = L_j, the buildings between are H_{L_j+1} ... H_{j-1}, all of which are < H_j.
    # So i can range from L_j to j-1.
    # The number of such i is (j-1) - L_j + 1 = j - L_j.
    # But we need to find for each i, the count of j.
    # This is a range update problem: for each j, increment range [L_j, j-1].
    
    # To solve this without loops:
    # 1. Find L_j for all j using a stack (via reduce).
    # 2. Use a difference array to perform range updates.
    # 3. Use prefix sums (via reduce) to get final counts.

    def find_l(acc, curr):
        stack, l_vals = acc
        h_curr, idx_curr = curr
        # Remove elements from stack that are smaller than current height
        # Since we can't use while, we use a helper function with reduce or a recursive-like structure
        # But we can't use recursion deeply. 
        # Actually, we can use a trick with a list and a custom function.
        # But the constraint is NO loops. 
        # Let's use a different approach for L_j.
        return acc

    # Since we cannot use while/for, we use a functional approach to find L_j.
    # We can use a divide and conquer approach or a Segment Tree, but that's hard without loops.
    # Let's use the property that we can use map/filter/reduce.
    # To find the nearest greater element to the left:
    # We can use a recursive function with a decorator for memoization or 
    # a technique using reduce to maintain the stack.
    
    def get_l_values(heights):
        # Using reduce to maintain a stack. 
        # To simulate the 'while' loop of the stack, we can use a helper 
        # that filters the stack. But filter is O(N), making it O(N^2).
        # However, we can use a technique where we store the stack and 
        # use a function to find the first element > H_j.
        # Given the constraints and the "no loop" rule, the most "functional" 
        # way to find nearest greater is using a Segment Tree or 
        # a Divide and Conquer approach implemented via recursion.
        
        def solve_recursive(l, r):
            if l == r:
                return [0] * 1 # Not used for L_j calculation directly
            # This is getting complex. Let's use the property that 
            # we can use list comprehensions (which are loops) ? 
            # No, the prompt says "no for or while loops". 
            # List comprehensions are technically loops.
            pass

    # Re-evaluating: the most efficient way to find L_j is a stack.
    # To implement a stack without while/for, we can use a recursive function.
    # Python's recursion limit is an issue, but we can increase it.
    sys.setrecursionlimit(300000)
    
    def find_l_recursive(h, idx, stack):
        if idx == len(h):
            return []
        # Find the first index in stack where h[stack[-1]] > h[idx]
        def pop_stack(s):
            if not s or h[s[-1]] > h[idx]:
                return s
            return pop_stack(s[:-1])
        
        new_stack = pop_stack(stack)
        l_val = new_stack[-1] + 1 if new_stack else 1
        return [l_val] + find_l_recursive(h, idx + 1, new_stack + [idx])

    # The above is still O(N^2) in worst case due to s[:-1].
    # Let's use a different approach. 
    # For each j, we want count of i < j such that max(H_{i+1}...H_{j-1}) < H_j.
    # This is equivalent to: j is visible from i.
    # This is true if i >= L_j, where L_j is the index of the first building 
    # to the left of j that is taller than H_j.
    # The number of such i is j - L_j.
    # We need to output for each i, the count of j.
    # This is sum_{j=i+1}^N [L_j <= i < j].
    
    # Let's use the property: j is visible from i if H_j > max(H_{i+1}...H_{j-1}).
    # This is exactly the number of elements in the "Right-side" 
    # monotonic stack starting from i.
    
    # Correct approach:
    # For a fixed i, the buildings j that satisfy the condition are the 
    # sequence of buildings that form a strictly increasing sequence of heights.
    # Example: 2 1 4 3 5
    # i=1 (H=2): j=2(H=1), j=3(H=4), j=5(H=5). Count=3.
    # i=2 (H=1): j=3(H=4), j=5(H=5). Wait, j=3 is 4, j=4 is 3. 
    # For i=2, j=3(H=4) is visible. Then j=4(H=3) is NOT because H_3=4 > 3.
    # Then j=5(H=5) is visible because H_3=4, H_4=3 are both < 5.
    # So for i=2, j=3 and j=5 are visible. Count=2.
    # Wait, the sample says for i=2, the answer is 2. My manual trace matches.
    
    # This is equivalent to: for each i, count j > i such that 
    # H_j is a prefix maximum of the array H[i+1:].
    # This is a known problem. The answer for i is the number of 
    # elements in the monotonic stack when processing H[i+1:] from left to right.
    
    # To solve this for all i without loops:
    # We can use a Segment Tree to find the number of prefix maximums.
    # For a range, the number of prefix maximums is:
    # count(left_child) + count_prefix_max(right_child, max_of_left_child)
    
    def build_tree(l, r):
        if l == r:
            return {'max': h[l], 'cnt': 1, 'l': l, 'r': r}
        mid = (l + r) // 2
        left = build_tree(l, mid)
        right = build_tree(mid + 1, r)
        return {
            'max': max(left['max'], right['max']),
            'cnt': left['cnt'] + count_greater(right, left['max']),
            'left': left,
            'right': right,
            'l': l,
            'r': r
        }

    def count_greater(node, threshold):
        if node['l'] == node['r']:
            return 1 if node['max'] > threshold else 0
        if node['left']['max'] <= threshold:
            return count_greater(node['right'], threshold)
        return (node['cnt'] - node['left']['cnt']) + count_greater(node['left'], threshold)

    # The above logic for count_greater is slightly wrong.
    # Correct logic for prefix maximums in range [L, R] given a threshold from the left:
    # If max(left_child) <= threshold:
    #    return count_greater(right_child, threshold)
    # Else:
    #    return count_greater(left_child, threshold) + (node.cnt - left_child.cnt)
    
    def count_pref(node, threshold):
        if node['l'] == node['r']:
            return 1 if node['max'] > threshold else 0
        if node['left']['max'] <= threshold:
            return count_pref(node['right'], threshold)
        # If left max > threshold, some elements in left are prefix maxes,
        # and all elements that were prefix maxes in the right child 
        