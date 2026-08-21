import sys

def solve():
    # Use fast I/O to handle N up to 2*10^5
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that Building j is a "visible" building when looking from i.
    # Specifically, j satisfies the condition if for all k such that i < k < j, H_k < H_j.
    # This is equivalent to saying that H_j is a new maximum encountered 
    # as we iterate from j-1 down to i+1.
    # However, a simpler way to think about it:
    # For a fixed i, we are looking for the number of j > i such that 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # Note that for j = i + 1, the set of buildings between i and j is empty,
    # so the condition is vacuously true.
    
    # This problem can be solved efficiently using a monotonic stack.
    # We want to count for each i, how many j > i satisfy the condition.
    # Let's process the buildings from right to left.
    # When we are at building i, the buildings j > i that satisfy the condition
    # are exactly the elements of a strictly increasing subsequence starting from 
    # the first building to the right (i+1) and picking every building that is 
    # taller than all buildings between i and it.
    
    # Actually, the condition "no building taller than Building j between i and j"
    # means H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means if we maintain a monotonic stack of buildings to the right of i,
    # the buildings that satisfy the condition are the ones that would be 
    # "visible" from the left.
    
    # Let's re-evaluate: for a fixed i, j satisfies the condition if H_j is a 
    # "prefix maximum" of the sequence H_{i+1}, H_{i+2}, ..., H_N.
    # Wait, that's not correct. The condition is:
    # For j = i + 1: always true.
    # For j = i + 2: true if H_{i+1} < H_{i+2}.
    # For j = i + 3: true if max(H_{i+1}, H_{i+2}) < H_{i+3}.
    # This means we are looking for the number of elements in the sequence 
    # H_{i+1}, ..., H_N that are strictly greater than all preceding elements 
    # in that specific subsequence.
    
    # This is exactly the number of elements that would remain in a monotonic 
    # stack if we processed the array from i+1 to N.
    # However, doing this for every i would be O(N^2). We need O(N log N) or O(N).
    
    # Observation: A building j satisfies the condition for i if H_j is 
    # greater than all H_k for i < k < j.
    # This means j is the "Next Greater Element" of some building, or it is i+1.
    # Let's use a stack to find for each j, the range of i's for which it is visible.
    # Building j is visible from i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # Let L[j] be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L[j] = 0.
    # Then building j is visible from i if i >= L[j] and i < j.
    # Wait, if L[j] is the index of the first building to the left taller than H_j,
    # then for any i such that L[j] <= i < j, the buildings between i and j 
    # are all shorter than H_j.
    # Let's check: if i = L[j], the buildings between i and j are H_{L[j]+1}...H_{j-1}.
    # By definition of L[j], all these are < H_j. So j is visible from L[j].
    # If i > L[j], the range is even smaller, so j is still visible.
    # If i < L[j], then H_{L[j]} is between i and j, and H_{L[j]} > H_j,
    # so j is NOT visible from i.
    
    # Therefore, for a fixed j, the indices i that satisfy the condition are:
    # i \in {L[j], L[j]+1, ..., j-1}.
    # But we must also have 1 <= i <= N.
    # So for each j from 2 to N, it contributes 1 to the count c_i for all i from L[j] to j-1.
    # Special case: Building j=1 cannot be "j" because i < j.
    # For j = 2 to N:
    #   Find L[j] = index of first k < j such that H_k > H_j.
    #   If no such k, L[j] = 1 (since i must be >= 1).
    #   Wait, if L[j] is the index of the first building taller than H_j,
    #   then for any i from L[j] to j-1, the condition is satisfied.
    #   Wait, if i = L[j], the buildings between i and j are H_{L[j]+1}...H_{j-1}.
    #   All these are < H_j. So j is visible from L[j].
    #   If i = L[j]-1, the building H_{L[j]} is between i and j, and H_{L[j]} > H_j.
    #   So j is NOT visible from L[j]-1.
    #   Thus, j is visible from i if L[j] <= i < j.
    #   Exception: if L[j] doesn't exist (no building to the left is taller),
    #   then j is visible from any i < j. So i \in {1, ..., j-1}.
    
    # Let's refine L[j]:
    # L[j] = index of the first building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 1.
    # Then j is visible from i for i = L[j], L[j]+1, ..., j-1.
    # Wait, if L[j] exists, then at i = L[j], the buildings between i and j are
    # H_{L[j]+1}, ..., H_{j-1}. All these are < H_j. So j is visible from L[j].
    # If i = L[j]-1, H_{L[j]} is between i and j, and H_{L[j]} > H_j.
    # So j is not visible from L[j]-1.
    # So the range of i is [L[j], j-1].
    # If L[j] doesn't exist, the range is [1, j-1].
    
    # Example 1: 2 1 4 3 5
    # j=2: H=1. L[2]=1 (H_1=2 > 1). i in [1, 1]. c_1++
    # j=3: H=4. L[3]=none. i in [1, 2]. c_1++, c_2++
    # j=4: H=3. L[4]=3 (H_3=4 > 3). i in [3, 3]. c_3++
    # j=5: H=5. L[5]=none. i in [1, 4]. c_1++, c_2++, c_3++, c_4++
    # c_1 = 1+1+1 = 3
    # c_2 = 1+1 = 2
    # c_3 = 1+1 = 2
    # c_4 = 1 = 1
    # c_5 = 0
    # Result: 3 2 2 1 0. Matches sample!
    
    L = [0] * N
    stack = [] # store indices
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if not stack:
            L[j] = 0 # No taller building to the left
        else:
            L[j] = stack[-1] # Index of first taller building
        stack.append(j)
        
    # Now we need to count for each i, how many j > i satisfy L[j] <= i < j.
    # This is a range update problem: for each j, increment [L[j], j-1].
    # We can use a difference array.
    diff = [0] * (N + 1)
    for j in range(1, N): # j starts from 2nd building (index 1)
        # Range of i is [L[j], j-1] (0-indexed)
        # If L[j] was -1, it would be 0. But L[j] is the index of the taller building.
        # Actually, based on the logic:
        # If L[j] exists (stack not empty), i can be L[j], L[j]+1, ..., j-1.
        # If L[j] doesn't exist, i can be 0, 1, ..., j-1.
        start = L[j] if L[j] != -1 else 0 # This is handled by the logic above
        # Wait, in my code L[j] is the index of the taller building.
        # If stack was empty, L[j] = 0. But that's the same as the first building.
        # Let's re-do the stack logic carefully.
    
    # Corrected Stack Logic:
    L = [0] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if not stack:
            L[j] = 0 # No taller building to the left, visible from all i < j
        else:
            L[j] = stack[-1] # Taller building at index L[j], visible from i >= L[j]
        stack.append(j)
        
    diff = [0] * (N + 1)
    for j in range(1, N):
        # Building j is visible from i in [L[j], j-1]
        # L[j] is the index of the first building to the left taller than H[j].
        # If no such building exists, L[j] = 0.
        # The range is i = L[j] to j-1.
        diff[L[j]] += 1
        diff[j] -= 1
        
    ans = []
    current = 0
    for i in range(N):
        current += diff[i]
        ans.append(current)
        
    print(*(ans))

if __name__ == "__main__":
    solve()