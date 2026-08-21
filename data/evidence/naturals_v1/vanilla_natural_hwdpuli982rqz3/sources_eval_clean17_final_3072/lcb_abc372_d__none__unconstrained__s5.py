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
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "right-to-left" 
    # record relative to Building i.
    # More simply: if we process the array from right to left, for a fixed i,
    # we want to count how many j > i are "visible" looking right.
    # A building j is visible from i if it is taller than all buildings between i and j.
    # This is exactly the number of elements in a monotonic increasing stack 
    # ( own height > previous max) when traversing from i+1 to N.
    
    # However, the constraint N=2e5 suggests an O(N log N) or O(N) solution.
    # Let's use the property: for a fixed i, the sequence of indices j that satisfy 
    # the condition are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2] ... and so on.
    # This is because any building between j1 and j2 is shorter than H[j1], 
    # and since H[j1] is between i and j2, for j2 to be valid, H[j2] must be > H[j1].
    
    # We can precompute the "Next Greater Element" (NGE) for every index.
    # Let next_greater[i] be the index of the first building to the right of i that is taller than H[i].
    # The number of valid j's for i is: 1 (for j=i+1) + count(next_greater[i+1])
    # But wait, the first building is always j = i+1. Then the next valid one is the 
    # first building taller than H[i+1], then the first building taller than that, etc.
    # This forms a chain.
    
    # Let dp[i] be the number of valid j's for index i.
    # For i = N-1 (0-indexed), dp[N-1] = 0.
    # For i < N-1:
    # The first valid j is i+1.
    # The subsequent valid j's are the ones that would be valid for i+1, 
    # BUT only those that are taller than H[i+1].
    # Actually, the condition is: H[k] < H[j] for i < k < j.
    # Let j1 = i + 1. j1 always satisfies this (no k between i and i+1).
    # For j > i + 1, the condition is: H[k] < H[j] for all k in {i+1, ..., j-1}.
    # This means H[j] must be greater than max(H[i+1], ..., H[j-1]).
    # This is exactly the definition of the sequence of "record" heights starting from i+1.
    
    # Let f(i) be the number of records in the suffix H[i:].
    # This is not quite right because the record sequence depends on the starting point.
    # Let's use the NGE approach:
    # For index i, the first valid j is i+1.
    # The next valid j is the first index k > i+1 such that H[k] > H[i+1].
    # The next is the first index k > that such that H[k] > H[that], and so on.
    # Let next_greater[i] be the index of the first element to the right of i that is > H[i].
    # Let count[i] be the number of elements in the chain starting at i.
    # count[i] = 1 + count[next_greater[i]] (if next_greater[i] exists, else 1).
    
    # To find next_greater for all i:
    next_greater = [N] * N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            next_greater[stack.pop()] = i
        stack.append(i)
        
    # To find dp[i]:
    # For a fixed i, the first j is i+1.
    # The number of valid j's is 1 + (number of records in H[i+1:] starting with H[i+1]).
    # Let g[i] be the number of records in the sequence H[i], H[next_greater[i]], H[next_greater[next_greater[i]]], ...
    # g[i] = 1 + g[next_greater[i]] if next_greater[i] < N else 1.
    
    g = [0] * (N + 1)
    # Compute g from right to left
    for i in range(N - 1, -1, -1):
        nxt = next_greater[i]
        if nxt < N:
            g[i] = 1 + g[nxt]
        else:
            g[i] = 1
            
    # For each i from 0 to N-2, the answer is g[i+1].
    # For i = N-1, the answer is 0.
    results = [g[i+1] if i < N-1 else 0 for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()