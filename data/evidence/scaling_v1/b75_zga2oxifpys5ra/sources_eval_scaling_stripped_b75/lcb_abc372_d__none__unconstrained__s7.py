import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to counting elements in the sequence H[i+1...N-1]
    # that are strictly greater than all preceding elements in that subsequence.
    # However, the condition is actually simpler: j satisfies the condition if
    # H[j] is a "right-side" visible building from i.
    # Specifically, j satisfies the condition if for all k such that i < k < j, H[k] < H[j].
    
    # Let's use a monotonic stack approach. 
    # For a fixed i, the buildings j that satisfy the condition are those that 
    # would remain in a monotonic increasing stack if we processed H[i+1...N-1].
    # But we need this for all i.
    
    # Correct observation:
    # For a fixed i, the indices j > i that satisfy the condition are:
    # 1. j = i + 1
    # 2. The index of the first building to the right of i+1 that is taller than H[i+1]
    # 3. The index of the first building to the right of that one that is taller than it, and so on.
    # Wait, that's not correct. The condition is: no building between i and j is taller than H[j].
    # This means H[j] > max(H[i+1], ..., H[j-1]).
    # This is exactly the definition of "left-to-right maxima" of the suffix starting at i+1.
    
    # Let f(i) be the number of j > i such that H[j] > max(H[i+1]...H[j-1]).
    # Let next_greater[k] be the index of the first building to the right of k taller than H[k].
    # The indices j are: (i+1), next_greater[i+1], next_greater[next_greater[i+1]], ...
    # The number of such indices is the distance to the end of the chain.
    
    # 1. Find next_greater array using a stack
    # next_greater[k] = N if no such building exists
    next_greater = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            next_greater[stack.pop()] = idx
        stack.append(idx)
        
    # 2. Use dynamic programming to count the chain length
    # dp[k] = 1 + dp[next_greater[k]]
    # We process from N-1 down to 0.
    dp = [0] * (N + 1)
    # Using a loop to avoid recursion depth issues for DP
    for k in range(N - 1, -1, -1):
        nxt = next_greater[k]
        dp[k] = 1 + (dp[nxt] if nxt < N else 0)
        
    # For each i, the answer is dp[i+1] (if i+1 < N)
    results = [dp[i+1] if i+1 < N else 0 for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()