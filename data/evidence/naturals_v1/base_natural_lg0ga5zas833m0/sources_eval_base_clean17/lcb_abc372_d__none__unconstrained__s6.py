import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition is: Building j satisfies the condition for Building i if
    # for all k such that i < k < j, H_k < H_j.
    # This means Building j must be a "right-side visible" building from i.
    # Specifically, if we iterate from i+1 to N, Building j is counted if 
    # H_j is greater than all heights encountered since Building i.
    # However, the condition is simpler: Building j is counted if H_j is the 
    # maximum height in the range [i+1, j].
    
    # Let's re-read: "There is no building taller than Building j between Buildings i and j."
    # This means for all k: i < k < j, H_k < H_j.
    # This is equivalent to saying H_j > max(H_{i+1}, ..., H_{j-1}).
    
    # For a fixed i, as we move j from i+1 to N:
    # j = i+1 always satisfies (range is empty).
    # j = i+2 satisfies if H_{i+1} < H_{i+2}.
    # j = i+3 satisfies if max(H_{i+1}, H_{i+2}) < H_{i+3}.
    # In general, j satisfies the condition if H_j is a new prefix maximum 
    # of the sequence H_{i+1}, H_{i+2}, ..., H_N.
    
    # To solve this efficiently for all i, we can use a Monotonic Stack.
    # We want to count how many j > i are "visible" from i.
    # A building j is visible from i if it is a prefix maximum of the suffix starting at i+1.
    # This is a classic problem that can be solved by processing from right to left.
    
    # Let's use a Segment Tree or a similar structure? No, that's too slow.
    # Let's use a Monotonic Stack to find the "next greater element".
    # For a fixed i, the visible buildings are:
    # j1 = i+1
    # j2 = the index of the first building in [j1+1, N] such that H_{j2} > H_{j1}
    # j3 = the index of the first building in [j2+1, N] such that H_{j3} > H_{j2}
    # ... and so on.
    
    # Let next_greater[i] be the index of the first building j > i such that H_j > H_i.
    # If no such j exists, next_greater[i] = N + 1.
    next_greater = [n + 1] * (n + 1)
    stack = []
    for i in range(n):
        # Use 1-based indexing for consistency with problem description
        # h[i] is height of building i+1
        while stack and h[stack[-1]] < h[i]:
            next_greater[stack[-1] + 1] = i + 1
            stack.pop()
        stack.append(i)
        
    # Now, for each i, the number of j's is the length of the chain:
    # i+1 -> next_greater[i+1] -> next_greater[next_greater[i+1]] ...
    # We can use dynamic programming to calculate this.
    # dp[i] = 1 + dp[next_greater[i]]
    dp = [0] * (n + 2)
    for i in range(n, 0, -1):
        nxt = next_greater[i]
        if nxt <= n:
            dp[i] = 1 + dp[nxt]
        else:
            dp[i] = 1 if i <= n else 0
            
    # The answer for building i is the number of visible buildings starting from i+1.
    # If i == N, c_N = 0.
    # If i < N, c_i = dp[i+1].
    # Wait, the DP logic: 
    # For i=1, the sequence is j1=2, j2=next_greater[2], j3=next_greater[j2]...
    # The number of elements is exactly dp[2].
    
    results = []
    for i in range(1, n + 1):
        if i < n:
            results.append(dp[i+1])
        else:
            results.append(0)
            
    print(*(results))

if __name__ == "__main__":
    solve()