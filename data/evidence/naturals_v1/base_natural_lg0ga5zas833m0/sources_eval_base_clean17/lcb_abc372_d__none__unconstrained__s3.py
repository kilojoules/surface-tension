import sys

def solve():
    # Increase recursion depth just in case, though not needed for this iterative solution
    sys.setrecursionlimit(300000)
    
    # Read all input at once for speed
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition: Building j (j > i) is counted if for all k such that i < k < j, H_k <= H_j.
    # This means Building j must be a "right-side" record relative to building i.
    # However, the condition is simpler: j satisfies the condition if H_j is greater than 
    # all heights in the range (i, j).
    # 
    # Let's rephrase: for a fixed i, we are looking for j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # 
    # This is equivalent to finding the number of elements in the sequence H_{i+1}, ..., H_N
    # that are strictly greater than all elements appearing before them in that subsequence.
    # 
    # For a fixed i, the buildings j that satisfy the condition are the "prefix maximums"
    # of the array starting from index i+1.
    # 
    # Since N is up to 2*10^5, an O(N^2) approach is too slow. We need something faster.
    # Note: The problem asks for this for every i.
    # 
    # Let's use a Monotonic Stack.
    # For a fixed j, for which i is it a "visible" building?
    # Building j is visible from i if for all k in (i, j), H_k < H_j.
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the condition is satisfied for all i such that L[j] <= i < j.
    # (Wait, the condition is: "no building taller than Building j between i and j").
    # If i = L[j], the buildings between i and j are indices L[j]+1 ... j-1.
    # By definition of L[j], all these are <= H_j. So i = L[j] is allowed.
    # If i < L[j], then building L[j] is between i and j, and H_{L[j]} > H_j, so the condition is violated.
    # So for a fixed j, the valid i's are i \in {L[j], L[j]+1, ..., j-1}.
    # However, the problem asks for i from 1 to N, and j > i.
    # So for a fixed j, the valid i's are i \in {max(1, L[j]), ..., j-1}.
    # 
    # Let's use a difference array (or Fenwick tree) to count this.
    # For each j from 2 to N:
    #   Find L[j] = index of nearest building to the left with H_{L[j]} > H_j.
    #   The range of i is [max(1, L[j]), j-1].
    #   Increment count for all i in this range.
    
    L = [0] * N
    stack = [] # stores indices
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1] + 1 # 1-based index
        else:
            L[j] = 0 # No taller building to the left
        stack.append(j)
    
    # Using a difference array to update ranges [max(1, L[j]), j]
    # Note: j is 0-indexed in H, so Building j is index j.
    # Range of i (1-based) is [max(1, L[j]), j] where j is 1-based index.
    # In 0-indexed terms for the result array:
    # Building j (index j) is visible from i (index i) if i < j and 
    # no building k (i < k < j) has H[k] > H[j].
    # This is true if i >= L[j] (where L[j] is the 0-indexed index of the first taller building to the left).
    # If no taller building exists, L[j] = -1.
    # So i can be L[j]+1, L[j]+2, ..., j-1. (Wait, if L[j] is the index, i can be L[j] itself).
    # Let's re-verify: i=1, j=3. Buildings between are index 2.
    # If H[2] < H[3], then j=3 is visible from i=1.
    # L[2] (for j=3) is the index of the first building to the left of index 2 taller than H[2].
    # Let's use the logic: j is visible from i if max(H_{i+1}...H_{j-1}) < H_j.
    # This means the first building to the left of j that is taller than H_j must be at index <= i.
    # Let prev_greater[j] be the index of the nearest building to the left of j with H[prev_greater] > H[j].
    # If no such building, prev_greater[j] = -1.
    # The condition "no building taller than Building j between i and j" means:
    # For all k: i < k < j, H[k] < H[j].
    # This is true if and only if prev_greater[j] <= i.
    # Since we also need i < j, the valid i's are: prev_greater[j] <= i < j.
    # But i must be at least 0 (0-indexed).
    # So i \in {max(0, prev_greater[j]), ..., j-1}.
    
    # Correct logic:
    # For each j from 1 to N-1:
    #   find pg = prev_greater[j]
    #   for i from max(0, pg) to j-1:
    #     count[i] += 1
    
    # To do this efficiently:
    diff = [0] * (N + 1)
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        pg = stack[-1] if stack else -1
        
        # Range of i is [max(0, pg), j-1]
        start = max(0, pg)
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
        stack.append(j)
        
    # Prefix sum of difference array to get actual counts
    results = []
    current = 0
    for i in range(N):
        current += diff[i]
        results.append(current)
        
    print(*(results))

if __name__ == '__main__':
    solve()