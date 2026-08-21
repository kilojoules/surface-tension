import sys
from functools import reduce

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking right from Building i.
    # A building j is visible from i if it is a left-to-right maximum 
    # of the subarray H[i+1...N-1].
    
    # However, the problem asks for this for every i.
    # Let's rephrase: j satisfies the condition for i if 
    # H[j] > max(H[i+1...j-1]) (with max of empty set being 0).
    # This means for a fixed j, it contributes to the count of all i < j
    # such that for all k in (i, j), H[k] < H[j].
    # This is true for all i from the index of the first building to the left 
    # of j that is taller than H[j], up to j-1.
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = -1.
    # Then j satisfies the condition for all i such that L[j] <= i < j.
    # Wait, the condition is "between i and j". 
    # If i = L[j], the buildings between are indices L[j]+1 ... j-1.
    # All these are shorter than H[j] by definition of L[j].
    # So i can range from L[j] to j-1.
    # The number of such i is j - L[j].
    # But we need the count for each i.
    # For a fixed i, we want the number of j > i such that max(H[i+1...j-1]) < H[j].
    
    # Let's use the property: j is counted for i if H[j] is a prefix maximum of H[i+1...N].
    # This is a classic problem that can be solved by processing the array from right to left
    # and maintaining a monotonic stack of indices of buildings that could be prefix maximums.
    # For a fixed i, the visible buildings j are those that form a strictly increasing 
    # subsequence starting from the first element to the right of i.
    
    # Using a monotonic stack:
    # For a fixed i, the visible buildings are:
    # 1. j = i + 1
    # 2. The first j > i + 1 such that H[j] > H[i+1]
    # 3. The first j > prev_j such that H[j] > H[prev_j]
    # and so on.
    
    # This structure is exactly what a Segment Tree or a sparse table combined with 
    # a recursive function can find, but since we need it for all i, 
    # we can observe that the "next greater element" (NGE) pointers form a forest.
    # Let next_greater[i] = min {j > i | H[j] > H[i]}.
    # The number of visible buildings for i is 1 + count(next_greater[i+1]) 
    # if i+1 < N, else 0.
    # Wait, the first visible building is always j = i + 1.
    # The next visible building is the first building to the right of i+1 that is taller than H[i+1].
    # That is exactly next_greater[i+1].
    # So c_i = 1 + f(i+1) where f(i) is the number of visible buildings starting from index i.
    # f(i) = 1 + f(next_greater[i]) if next_greater[i] exists, else 1.
    
    # 1. Find next_greater index for all i
    # We use a stack to find the next greater element in O(N).
    next_greater = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            next_greater[stack.pop()] = idx
        stack.append(idx)
    
    # 2. Calculate f(i) using dynamic programming from right to left
    # f[i] = 1 + f[next_greater[i]] if next_greater[i] < N else 1
    f = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        nxt = next_greater[i]
        if nxt < N:
            f[i] = 1 + f[nxt]
        else:
            f[i] = 1
            
    # 3. For each i, the answer is f[i+1] if i+1 < N else 0
    # Note: The problem asks for i = 1 to N. In 0-indexing, that's i = 0 to N-1.
    # For i, we look at buildings j from i+1 to N-1.
    # The number of such j is f[i+1].
    results = [f[i+1] if i+1 < N else 0 for i in range(N)]
    
    print(*(results))

if __name__ == "__main__":
    solve()