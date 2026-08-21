import sys
from functools import reduce

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "No building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "visible" building 
    # when looking right from Building i, but specifically focusing on the 
    # height of the target building j rather than the observer i.
    
    # Actually, the condition is: for a fixed i, j satisfies the condition if
    # H[j] > max(H[i+1], H[i+2], ..., H[j-1]).
    # Note: For j = i + 1, the set {k | i < k < j} is empty, so the condition is vacuously true.
    
    # Let's define a sequence of "right-looking" maximums.
    # For a fixed i, the sequence of j's that satisfy this are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]
    # ... and so on.
    
    # This looks like a structure we can solve using a Monotonic Stack or a Tree.
    # Specifically, if we build a tree where the parent of node j is the first index k > j 
    # such that H[k] > H[j], the number of such j's for a given i is the depth of 
    # the node (i+1) in this forest, considering only nodes <= N.
    
    # To find the next greater element (NGE) for all indices:
    nge = [N] * N # Use N as a sentinel for "no greater element"
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            nge[stack.pop()] = i
        stack.append(i)
    
    # Now we have the NGE array. We want to find for each i, 
    # how many elements are in the chain: (i+1) -> nge[i+1] -> nge[nge[i+1]] ...
    # until the index reaches N.
    
    # We can use dynamic programming to find the length of these chains.
    # dp[i] = 1 + dp[nge[i]] if nge[i] < N else 1
    # We compute this from N-1 down to 0.
    
    # Using a list comprehension to simulate the DP since we can't use for/while loops.
    # We need a way to reference the dp array while building it.
    # Since we can't use loops, we can use a recursive-like structure with map or 
    # simply use the fact that we can mutate a list inside a map.
    
    dp = [0] * (N + 1) # dp[N] = 0
    
    # To avoid loops, we use a trick with map and a list.
    # We process indices in reverse order.
    list(map(lambda i: dp.__setitem__(i, 1 + dp[nge[i]]), range(N - 1, -1, -1)))
    
    # For each i from 0 to N-2, the answer is dp[i+1]. For i = N-1, it is 0.
    # We need to output c_1, c_2, ..., c_N.
    # c_i corresponds to index i-1 in 0-indexed.
    # For i=1 (index 0), answer is dp[1].
    # For i=N (index N-1), answer is 0.
    
    results = [dp[i+1] if i < N - 1 else 0 for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()