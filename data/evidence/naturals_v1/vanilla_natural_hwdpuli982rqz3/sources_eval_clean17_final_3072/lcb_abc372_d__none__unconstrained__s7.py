import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of j > i
    # such that H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum when looking at the range (i, N].
    # More simply: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means the sequence of buildings that satisfy the condition for a fixed i
    # are the indices of the prefix maximums of the array H starting from index i+1.
    
    # To solve this for all i efficiently, we can use a Segment Tree or a 
    # Divide and Conquer approach. However, a simpler observation is:
    # For a fixed j, it is counted for all i < j such that max(H[i+1...j-1]) < H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # Then for all i from L[j] to j-1, the condition is satisfied.
    # The number of such i is j - L[j]. (Using 0-based indexing).
    
    # We can find L[j] for all j using a monotonic stack.
    # L[j] = max({k | k < j and H[k] > H[j]} union {-1})
    
    L = [-1] * N
    stack = []
    
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        stack.append(j)
        
    # Now we have L[j]. For each j, it contributes 1 to c_i for i in [L[j], j-1].
    # We need to calculate c_i = sum_{j=i+1}^{N-1} [L[j] <= i]
    # This is a range update, point query problem.
    # We can use a difference array to handle the range updates.
    # For each j, we add 1 to range [L[j], j-1].
    
    diff = [0] * (N + 1)
    for j in range(N):
        # Range is [L[j], j-1]. Note: L[j] can be -1.
        # Since i ranges from 0 to N-1, we handle the boundary.
        start = L[j]
        end = j - 1
        # We want to increment c_i for i in [start, end]
        # But i must also be >= 0.
        # Since we can't have negative indices in diff, and L[j] is the 
        # index of the taller building, the buildings between L[j]+1 and j-1
        # are all shorter than H[j].
        # Actually, the condition is: j is counted for i if max(H[i+1...j-1]) < H[j].
        # This is true for i such that i+1 > L[j], i.e., i >= L[j].
        # Also i < j.
        # So i is in {L[j], L[j]+1, ..., j-1}.
        # Wait, if L[j] is the index of the first building taller than H[j],
        # then for i = L[j], the buildings between i and j are H[L[j]+1...j-1],
        # all of which are < H[j]. So i = L[j] is included.
        # For i = L[j]-1, the building at L[j] is between i and j, and H[L[j]] > H[j].
        # So i = L[j]-1 is NOT included.
        # Thus i ranges from L[j] to j-1.
        # Since we need i >= 0, the range is [max(0, L[j]), j-1].
        
        s = max(0, L[j])
        e = j - 1
        if s <= e:
            diff[s] += 1
            diff[e + 1] -= 1
            
    # Compute prefix sums to get c_i
    results = [0] * N
    current = 0
    for i in range(N):
        current += diff[i]
        results[i] = current
        
    print(*(results))

if __name__ == "__main__":
    solve()