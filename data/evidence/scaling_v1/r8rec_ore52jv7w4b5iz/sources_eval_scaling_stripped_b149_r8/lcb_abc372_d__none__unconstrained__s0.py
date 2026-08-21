import sys
from bisect import bisect_right

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # A building j is visible from i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that Building j is part of the 
    # sequence of prefix maximums of the array H[i+1:].
    
    # To solve this efficiently for all i, we can use a Divide and Conquer 
    # approach similar to counting inversions or use a Segment Tree/Fenwick Tree.
    # However, a simpler observation is:
    # Building j is counted for i if H_j is greater than all heights in range (i, j).
    # This is a classic problem that can be solved by processing 
    # buildings in decreasing order of height or using a Segment Tree.
    
    # Let's use a Divide and Conquer approach to count pairs (i, j) 
    # such that i < j and max(H_{i+1}...H_{j-1}) < H_j.
    
    # Actually, the condition is: j satisfies the condition if 
    # H_j > max(H_k) for all i < k < j.
    # This means for a fixed i, we are looking for the number of elements 
    # in the sequence H[i+1:] that are strictly greater than all preceding 
    # elements in that suffix.
    
    # This is equivalent to: for a fixed j, it is counted for all i < j 
    # such that max(H_{i+1}...H_{j-1}) < H_j.
    # Let L_j be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L_j = 0.
    # Then for any i such that L_j <= i < j, building j satisfies the condition.
    # (Note: if i = L_j, the buildings between i and j are all shorter than H_j).
    # So for each j, it contributes to i in the range [L_j, j-1].
    # The number of such i is j - L_j.
    
    # We need to find L_j for all j=1...N. This is the "Nearest Greater Element to the Left" problem.
    # We can solve this using a monotonic stack in O(N).
    
    # Since we need the count c_i for each i, and each j contributes to i in [L_j, j-1],
    # we can use a difference array (or Fenwick tree) to add 1 to range [L_j, j-1].
    # Then compute the prefix sum to get c_i.
    
    # Using 0-indexing for implementation:
    # Building j (0-indexed) contributes to i in range [L_j, j-1].
    # L_j is the index of the first k < j such that H_k > H_j. 
    # If no such k, L_j = 0 (since i can be 0, and buildings between 0 and j are H[1...j-1]).
    # Wait, if H_k > H_j, then for any i < k, building k is between i and j 
    # and H_k > H_j, so the condition is violated.
    # So i must be >= k. The range of i is [k, j-1].
    # If no such k exists, i can be any value from 0 to j-1.
    
    # Let's refine:
    # For a fixed j, we seek i < j such that max(H_{i+1}, ..., H_{j-1}) < H_j.
    # Let k be the index of the nearest element to the left of j such that H_k > H_j.
    # If such k exists, then for any i < k, H_k is between i and j and H_k > H_j, 
    # so the condition fails.
    # For i = k, the buildings between i and j are H_{k+1}...H_{j-1}, all of which are < H_j.
    # For i > k, the buildings between i and j are a subset of H_{k+1}...H_{j-1}, all < H_j.
    # So i can be k, k+1, ..., j-1.
    # If no such k exists, i can be 0, 1, ..., j-1.
    
    # Implementation:
    # 1. Find L_j = index of nearest element to the left > H_j.
    # 2. Use a difference array to mark ranges [L_j, j-1].
    # 3. Compute prefix sums.
    
    # To avoid loops, we use a stack-based approach to find L_j.
    # Since we can't use loops, we'll use a recursive-like structure or 
    # map/reduce, but the constraints allow a list comprehension 
    # if we can express the logic. Actually, the "Nearest Greater Element" 
    # can be solved without explicit loops using a trick with 
    # list comprehensions and a helper function, but it's tricky.
    # However, the prompt allows "complete Python program". 
    # Standard loops (for/while) are allowed in Python.
    
    stack = []
    l_bounds = [0] * n
    # We need to populate l_bounds using a loop. 
    # The constraint says "Return only Python source", it doesn't forbid loops.
    for j in range(n):
        while stack and h[stack[-1]] < h[j]:
            stack.pop()
        l_bounds[j] = stack[-1] + 1 if stack else 0
        stack.append(j)
    
    # Difference array to count c_i
    # range [L_j, j-1] gets +1
    diff = [0] * (n + 1)
    for j in range(n):
        # i is in [l_bounds[j], j-1]
        # Note: if j=0, the range is [0, -1], which is empty.
        if j > 0:
            diff[l_bounds[j]] += 1
            diff[j] -= 1
            
    # Compute prefix sums to get c_i
    # Using a list comprehension with a running total is hard without loops.
    # But we can use itertools.accumulate.
    from itertools import accumulate
    results = list(accumulate(diff))
    
    # We need c_1 to c_N. results[i] corresponds to c_i.
    # The diff array was size N+1, results is size N+1.
    # We need results[0] to results[n-1].
    print(*(results[:n]))

if __name__ == "__main__":
    solve()