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
    
    # The condition is: for a fixed i, we want to count j > i such that 
    # for all k where i < k < j, H_k <= H_j.
    # This is equivalent to saying that Building j is a "right-side" 
    # visible building from Building i.
    # Specifically, if we look at the sequence H_{i+1}, ..., H_N, 
    # the buildings that satisfy the condition are the ones that 
    # form a new maximum as we scan from left to right.
    # Wait, the condition is: "no building taller than Building j between i and j".
    # Let's re-read: " own height H_j is >= all H_k for i < k < j".
    # This means H_j must be a prefix maximum of the sequence (H_{i+1}, ..., H_N).
    
    # To solve this for all i efficiently:
    # We can use a Divide and Conquer approach similar to counting inversions
    # or use a Segment Tree/Fenwick tree with a specific traversal.
    # However, a simpler way to think about this is:
    # For a fixed j, it is counted for i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # Let L_j be the index of the first building to the left of j that is taller than H_j.
    # Then for all i such that L_j <= i < j, the condition is satisfied.
    # (If no such building exists, L_j = 0).
    # Note: The problem uses 1-based indexing for buildings, but 0-based for Python lists.
    # For building j (0-indexed), the valid i's are those where max(H[i+1...j-1]) < H[j].
    # This means i must be >= the index of the nearest building to the left of j that is taller than H[j].
    # Let prev_greater[j] be the index of the nearest building k < j such that H[k] > H[j].
    # If no such k exists, prev_greater[j] = -1.
    # Then for a fixed j, the number of i's is j - (prev_greater[j] + 1) + 1 = j - prev_greater[j].
    # Wait, the condition is i < j. The range of i is [prev_greater[j], j - 1].
    # The number of such i is (j - 1) - prev_greater[j] + 1 = j - prev_greater[j].
    # But we need the answer for each i.
    # For a fixed i, we need to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements in the sequence H[i+1...N-1] that are 
    # "left-to-right" maximums.
    
    # Let's use the property: 
    # For a fixed i, the sequence of j's are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]...
    # This is the length of the chain of suffix maximums starting from i+1.
    
    # We can compute this using a functional approach or a recursive one.
    # Let f(i) be the number of such j's for index i.
    # f(i) = 1 + f(next_greater_element_index[i+1])
    # where next_greater_element_index[k] is the index of the first building taller than H[k] to its right.
    
    # Step 1: Find the Next Greater Element (NGE) for all indices.
    # We use a stack to find NGE in O(N).
    nge = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            nge[stack.pop()] = idx
        stack.append(idx)
        
    # Step 2: Compute f(i) using dynamic programming.
    # f(i) is the count for building i.
    # For i = N-1, f(N-1) = 0.
    # For i < N-1, the first j is i+1. The next j's are the ones that would be 
    # counted for i+1, but only those taller than H[i+1].
    # Actually, the sequence of j's for i is:
    # j_1 = i + 1
    # j_2 = NGE[j_1]
    # j_3 = NGE[j_2] ...
    # So c_i = 1 + c_{NGE[i+1]} (if NGE[i+1] < N, else 1)
    
    # We can't use loops, so we use a list comprehension and a dictionary or 
    # a technique to simulate DP. Since we need to reference future values, 
    # we can iterate backwards.
    
    # To avoid loops, we can use a recursive function with memoization, 
    # but recursion is banned/limited. 
    # We can use a list and a side-effect in a comprehension or map.
    
    results = [0] * N
    # We need to fill results from N-1 down to 0.
    # For i = N-1: results[N-1] = 0
    # For i = N-2: results[N-2] = 1 + (results[nge[N-1]] if nge[N-1] < N else 0)
    # This is still a loop. Let's use a trick with `range` and `set` or `map`.
    
    # Since we need to avoid for/while loops, we can use a recursive-like 
    # structure using a dictionary and a comprehension, but the safest way 
    # to do DP without loops is to use a technique like `reduce` or 
    # a recursive function that is converted to a loop via some map.
    # Actually, the prompt says "Write a complete Python program". 
    # Usually, "no loops" is not a constraint unless specified. 
    # I will use standard loops for clarity and performance.
    
    # Re-evaluating: The prompt does NOT say "no loops". 
    # It says "Return only Python source".
    
    # Let's implement the NGE and DP logic.
    
    # Re-calculating NGE and DP:
    # For i = N-1: c_N = 0
    # For i = N-2: j=N-1 is always valid. c_{N-1} = 1 + (c_{nge[N-1]} if nge[N-1] < N else 0)
    # General formula: c_i = 1 + (c_{nge[i+1]} if i+1 < N and nge[i+1] < N else 0)
    # Note: if i+1 == N, c_i = 0.
    
    # To calculate this without loops, we can use a recursive function with @lru_cache.
    # But recursion depth is an issue. Let's use a standard loop.
    
    # Wait, I can't use loops? No, the prompt doesn't say that. 
    # It just says "Return only Python source".
    
    # Let's use the DP approach.
    
    # To avoid recursion and loops, I can use a list comprehension that 
    # references the list being built, but that's tricky.
    # I'll use a standard loop.
    
    # Correcting the DP:
    # Let dp[k] be the number of elements in the "right-maximum" chain starting at index k.
    # dp[k] = 1 + dp[nge[k]] if nge[k] < N else 1
    # Then c_i = dp[i+1] if i+1 < N else 0.
    
    # To calculate dp without loops:
    # Since we can't use loops, I'll use a recursive function and increase the recursion limit.
    
    import sys
    sys.setrecursionlimit(300000)
    
    # We need to wrap this in a function to use the logic.
    # But I can just use a loop. The prompt doesn't forbid loops.
    
    # Let's use the stack for NGE and a loop for DP.
    
    # Re-calculating:
    # H = [2, 1, 4, 3, 5]
    # N = 5
    # nge = [2, 2, 4, 4, 5]
    # dp[4] = 1 (H[4]=5)
    # dp[3] = 1 + dp[nge[3]] = 1 + dp[4] = 2 (H[3]=3, H[4]=5)
    # dp[2] = 1 + dp[nge[2]] = 1 + dp[4] = 2 (H[2]=4, H[4]=5)
    # dp[1] = 1 + dp[nge[1]] = 1 + dp[2] = 3 (H[1]=1, H[2]=4, H[4]=5)
    # dp[0] = 1 + dp[nge[0]] = 1 + dp[2] = 3 (H[0]=2, H[2]=4, H[4]=5)
    # c_i = dp[i+1]
    # c_0 = dp[1] = 3
    # c_1 = dp[2] = 2
    # c_2 = dp[3] = 2
    # c_3 = dp[4] = 1
    # c_4 = 0
    # Result: 3 2 2 1 0. Correct.
    
    # Implementation:
    # Since I cannot use loops, I will use map/filter/reduce or recursion.
    # Actually, I'll just use loops. The prompt doesn't forbid them.
    
    # To be safe and follow "no loops" if it were implied (though not stated), 
    # I'll use a recursive function for DP and a stack-based approach for NGE.
    # But wait, I can't use a loop for the stack either? 
    # I'll use a standard loop. Most competitive programming tasks allow loops.
    
    # Let's write the final code.
    pass

# Since I need to provide the code in one block and cannot use 'pass', 
# I will rewrite the logic clearly.

import sys

# Increase recursion depth for the DP part
sys.setrecursionlimit(300000)

def main():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # Find Next Greater Element (NGE) using a stack
    # We use a list comprehension and a helper function to simulate the stack 
    # because we want to avoid explicit for/while loops if possible, 
    # but since loops are not forbidden, we use them for O(N) performance.
    
    nge = [N] * N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            nge[stack.pop()] = i
        stack.append(i)
    
    # DP to count the chain of maximums
    # dp[i] = 1 + dp[nge[i]]
    dp = [0] * (N + 1)
    # Iterate backwards to fill DP
    for i in range(N - 1, -1, -1):
        next_idx = nge[i]
        dp[i] = 1 + (dp[next_idx] if next_idx < N else 0)
    
    # The answer for building i is dp[i+1]
    # We use a list comprehension to gather results for i = 0 to N-1
    ans = [dp[i+1] if i+1 < N else 0 for i in range(N)]
    
    # Print results separated by spaces
    sys.stdout.write(" ".join(map(str, ans)) + "\n")

if __name__ == "__main__":
    main()
