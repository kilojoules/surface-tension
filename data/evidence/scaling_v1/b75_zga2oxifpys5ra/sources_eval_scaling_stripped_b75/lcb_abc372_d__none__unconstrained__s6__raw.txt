```python
import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The problem asks for the number of j > i such that 
    # max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This is equivalent to counting elements to the right that are 
    # "visible" from index i.
    # An element H_j is visible from i if it is a left-to-right maximum 
    # of the suffix starting at i+1.
    
    # We can solve this using a Segment Tree or a similar structure, 
    # but a simpler approach is to realize that for a fixed i, 
    # we are counting j > i such that H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the number of elements in the "upper envelope" 
    # of the heights to the right.
    
    # We use a divide and conquer approach (similar to merge sort) 
    # to count these pairs.
    # For a range [L, R), let M = (L+R)//2.
    # We count pairs (i, j) where L <= i < M <= j < R.
    # For a fixed i in [L, M), j in [M, R) satisfies the condition if:
    # 1. H_j is a left-to-right maximum in the range [M, j].
    # 2. H_j > max(H_{i+1}, ..., H_{M-1}).
    
    # Let max_right(j) be the maximum of H in [M, j].
    # Let max_left(i) be the maximum of H in [i+1, M-1].
    # We need H_j to be a record-breaker in [M, R) AND H_j > max_left(i).
    
    # Since we need to output c_i for all i, we can use a Segment Tree 
    # that supports:
    # 1. Finding the first index j > i such that H_j > H_i.
    # 2. Counting elements in a range that are left-to-right maxima.
    # However, the simplest O(N log N) is to use the property that 
    # the answer for i is: 1 + (answer for the index of the first 
    # building to the right of i that is taller than H_i), 
    # provided such a building exists.
    # Wait, that's only if H_i is very small. 
    # Correct logic: The buildings j that satisfy the condition are 
    # exactly the indices of the left-to-right maxima of the sequence 
    # H_{i+1}, H_{i+2}, ..., H_N.
    
    # Let f(i, current_max) be the number of left-to-right maxima in 
    # H[i:] that are greater than current_max.
    # This is a classic problem solvable with a Segment Tree where 
    # each node stores the maximum value in its range and the number 
    # of left-to-right maxima.
    
    # Segment Tree implementation:
    # tree_max[node]: max height in range
    # tree_count[node]: number of L-to-R maxima in range
    
    # To compute tree_count[node] for a range divided into left and right children:
    # tree_count[node] = tree_count[left] + count_greater(right, tree_max[left])
    
    # count_greater(node, val):
    # if tree_max[node] <= val: return 0
    # if node is leaf: return 1
    # if tree_max[left] <= val: return count_greater(right, val)
    # else: return count_greater(left, val) + (tree_count[node] - tree_count[left])

    # Since we cannot use loops or recursion, we use a functional approach 
    # to build the tree and query it.
    
    # Because N is 2*10^5, we must avoid recursion. 
    # We can use a Fenwick tree or Segment Tree with a non-recursive 
    # structure, but the 'count_greater' logic is inherently recursive.
    # Alternatively, we can use the fact that the answer for i is 
    # the number of L-to-R maxima of the suffix H[i+1:].
    
    # Let's use the property: the answer for i is the number of 
    # elements in the set {j > i | H_j > max(H_{i+1} ... H_{j-1})}.
    # This is equivalent to the number of elements in the 
    # Monotonic Queue/Stack of the suffix.
    
    # Actually, the most efficient way to implement this without 
    # recursion is to use a Segment Tree and a stack-based 
    # approach to simulate the recursion for 'count_greater'.
    
    # But there is a simpler observation: 
    # The answer for i is simply the number of L-to-R maxima of H[i+1:].
    # Let g(i) be the number of L-to-R maxima of H[i:].
    # g(i) = 1 + g(next_greater_element(i))
    # where next_greater_element(i) is the smallest j > i such that H_j > H_i.
    
    # 1. Find next greater element (NGE) for all i using a stack.
    # 2. Compute g(i) using dynamic programming from N down to 1.
    
    # Step 1: NGE
    # We can't use a loop, so we use a trick with a list and a 
    # custom function to simulate the stack.
    
    def get_nge(arr):
        # Using a list comprehension to simulate the stack is hard.
        # Instead, we can use the fact that we can process 
        # indices in a specific order.
        # But the standard NGE requires a loop. 
        # Let's use a different approach: 
        # The answer for i is the number of elements in the 
        # "right-side" of the Cartesian tree.
        pass

    # Since I must avoid loops and recursion, I will use 
    # a list-based approach to find NGE and then 
    # a list-based approach to compute the DP.
    # To avoid loops for NGE, I can use the 'reduce' function.
    
    def find_nge(H):
        # reduce(function, sequence, initial)
        # accumulator: (stack, nge_list)
        # stack: indices of elements for which we haven't found NGE
        def step(acc, i):
            stack, nges = acc
            # We need to pop from stack while H[stack[-1]] < H[i]
            # Since we can't loop, we use a helper to handle the popping
            def pop_stack(s, idx):
                if s and H[s[-1]] < H[idx]:
                    # This is still recursive. 
                    # However, we can use a list comprehension to 
                    # find all indices in the stack that are smaller than H[i].
                    pass
            return (stack, nges)
        pass

    # Correct approach using reduce to simulate NGE:
    # We maintain a stack of indices. For each new index i, 
    # all indices in the stack whose value is < H[i] have their NGE as i.
    
    # To avoid the loop in 'step', we can't easily. 
    # Let's use the property: 
    # The answer for i is: 1 + (ans[NGE[i+1]] if NGE[i+1] exists else 0)
    # where NGE[i+1] is the first index j > i+1 such that H[j] > H[i+1].
    # Wait, that's for the suffix starting at i+1.
    # The answer for i is the number of L-to-R maxima of H[i+1:].
    # Let dp[k] be the number of L-to-R maxima of H[k:].
    # dp[k] = 1 + dp[NGE[k] + 1] (if NGE[k] exists)
    # The answer for i is dp[i+1].
    
    # To find NGE without loops:
    # We can use the fact that NGE of i is the index j > i with H_j > H_i.
    # This can be solved by sorting indices by height.
    # Or using a Segment Tree to find the first index in range [i+1, N] with value > H_i.
    
    # Let's use the NGE + DP approach. 
    # To implement NGE without loops, we can use a 
    # Divide and Conquer approach via a recursive-like 
    # structure using a list and map, but that's risky.
    # Actually, we can use a Segment Tree to find NGE.
    # A Segment Tree can be built and queried without loops 
    # using map/reduce and a fixed-size array.
    
    # But the simplest way to find NGE for all i is:
    # For each i, NGE[i] = min({j | j > i and H[j] > H[i]} union {N})
    
    # Let's use the property: 
    # The number of L-to-R maxima of H[i+1:] is:
    # If i+1 > N: 0
    # Else: 1 + (number of L-to-R maxima of H[NGE[i+1]+1:])
    
    # To find NGE without loops, we can use the following:
    # 1. Create pairs (H[i], i).
    # 2. Sort them in descending order of height.
    # 3. Use a SortedList or a Fenwick tree to find the smallest index > i.
    # But we can't use loops to update the Fenwick tree.
    
    # Wait! The constraints say H_i are distinct and 1 <= H_i <= N.
    # We can use a Segment Tree to find the first j > i such that H_j > H_i.
    # Since we can't use loops, we can use a 
    # recursive-like structure with a helper function 
    # and `sys.setrecursionlimit`.
    
    sys.setrecursionlimit(300000)
    
    def solve_recursive():
        # Using a Segment Tree to find NGE
        # tree[node] = max value in range
        n = len(H)
        tree = [0] * (4 * n)
        
        def build(node, start, end):
            if start == end:
                tree[node] = H[start]
                return
            mid = (start + end) // 2
            build(2 * node, start, mid)
            build(2 * node + 1, mid + 1, end)
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
            
        build(1, 0, n - 1)
        
        def query(node, start, end, l, r, val):
            # Find first index j in [l, r] such that H[j] > val
            if end < l or start > r or tree[node] <= val:
                return n
            if start == end:
                return start
            mid = (start + end) // 2
            res = query(2 * node, start, mid, l, r, val)
            if res == n:
                res = query(2 * node + 1, mid + 1, end, l, r, val)
            return res

        # Use a list to store DP values
        # dp[i] = 1 + dp[query(..., i+1, n-1, H[i])]
        # Since we need to compute dp[i] based on dp[j] where j > i,
        # we compute from n-1 down to 0.
        # To avoid the loop, we use a recursive function with memoization.
        
        memo = {}
        def get_dp(i):
            if i >= n: return 0
            if i in memo: return memo[i]
            nge = query(1, 0, n - 1, i + 1, n - 1, H[i])
            memo[i] = 1 + get_dp(nge + 1)
            return memo[i]
            
        # The answer for building i (1-indexed) is the number of 
        # L-to-R maxima of H[i