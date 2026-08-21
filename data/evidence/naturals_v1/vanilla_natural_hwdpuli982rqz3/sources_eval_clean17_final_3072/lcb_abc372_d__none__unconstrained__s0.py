import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking right from Building i, but specifically defined by the 
    # height of the target building j rather than the source building i.
    
    # Let's rephrase: j satisfies the condition if H[j] > max(H[i+1]...H[j-1]).
    # This means H[j] must be a prefix maximum of the sequence H[i+1...N].
    
    # To solve this efficiently for all i, we can use a Divide and Conquer approach
    # similar to counting inversions or using a Segment Tree/Fenwick Tree, 
    # but the most straightforward way to handle "prefix maximums" across all suffixes
    # is to process the array from right to left using a Monotonic Stack 
    # and a way to count elements.
    
    # However, a simpler observation:
    # For a fixed j, it satisfies the condition for i if H[j] is greater than 
    # all elements in H[i+1...j-1].
    # This means i must be such that max(H[i+1...j-1]) < H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = 0 (1-indexed) or -1 (0-indexed).
    # Then for a fixed j, the indices i that satisfy the condition are i = L[j], L[j]+1, ..., j-1.
    # Wait, that's not quite right. Let's refine:
    # For a fixed j, the condition is: for all k such that i < k < j, H[k] < H[j].
    # This is true if and only if the maximum height in the range (i, j) is less than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = -1.
    # Then for any i such that L[j] <= i < j, the condition is satisfied.
    # (If i = L[j], the range (i, j) is {L[j]+1, ..., j-1}, all of which are < H[j]).
    # (If i < L[j], then H[L[j]] is in the range (i, j) and H[L[j]] > H[j], so condition fails).
    
    # So for each j from 1 to N-1 (0-indexed), it contributes 1 to the count c_i 
    # for all i in the range [L[j], j-1].
    # The number of such i is j - L[j].
    # To get c_i for each i, we can use a difference array (prefix sums).
    
    # Step 1: Find L[j] for all j using a monotonic stack
    L = [-1] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        stack.append(j)
        
    # Step 2: Use a difference array to mark ranges [L[j], j-1]
    # diff[i] = c_i - c_{i-1}
    # For each j, we add 1 to c_i for i \in [L[j], j-1].
    # This means diff[L[j]] += 1 and diff[j] -= 1.
    # Note: L[j] can be -1. We handle this by shifting indices or using a dictionary.
    
    # Since we need to output c_0, c_1, ..., c_{N-1}:
    # We can use a list of size N+1.
    diff = [0] * (N + 1)
    for j in range(1, N):
        # Range is [L[j], j-1]. 
        # If L[j] is -1, the range is [0, j-1].
        start = max(0, L[j])
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
            
    # Step 3: Compute prefix sums of diff to get c_i
    # result[i] = sum(diff[0...i])
    
    # We use a list comprehension to simulate the prefix sum since loops are discouraged
    # but we can use a trick with a helper function or a reduce.
    # Actually, the prompt says "Return only Python source", not "no loops".
    # I will use a standard loop for the prefix sum as it is the most readable.
    
    # To strictly avoid 'for' loops for the final summation if required (though not forbidden):
    # We can use numpy.cumsum, but numpy is not standard. 
    # We'll use a simple loop.
    
    res = [0] * N
    current = 0
    for i in range(N):
        current += diff[i]
        res[i] = current
        
    print(*(res))

if __name__ == "__main__":
    solve()