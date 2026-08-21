```python
import sys

def solve():
    # Increase recursion depth for deep stacks if necessary, 
    # though we use an iterative approach here.
    sys.setrecursionlimit(300000)
    
    # Read N from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The problem asks for the number of j > i such that 
    # for all k (i < k < j), H_k <= H_j.
    # This is equivalent to counting how many elements to the right of i
    # are "visible" if we only consider elements that are strictly greater 
    # than all elements appearing between i and that element.
    
    # Actually, the condition "There is no building taller than Building j 
    # between Buildings i and j" means that H_j must be the maximum height 
    # in the range [i+1, j].
    
    # Let's rephrase: for a fixed i, we are looking for j > i such that
    # H_j = max(H_{i+1}, H_{i+2}, ..., H_j).
    # This means the sequence of heights H_j that satisfy this condition 
    # for a fixed i are the "prefix maximums" of the sequence H_{i+1}, ..., H_N.
    
    # However, N is up to 2*10^5, so an O(N^2) approach will be too slow.
    # We need a more efficient way.
    
    # Let's observe: for a fixed j, for which i < j is the condition satisfied?
    # The condition is: for all k such that i < k < j, H_k <= H_j.
    # This means i must be such that there is no k in (i, j) where H_k > H_j.
    # Let L_j be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L_j = 0.
    # Then for any i such that L_j <= i < j, the condition is satisfied.
    # (Wait, the condition is "between i and j", so i < k < j. 
    # If i = L_j, the buildings between i and j are indices L_j + 1, ..., j-1.
    # By definition of L_j, all these are <= H_j. So i = L_j is allowed.
    # If i < L_j, then k = L_j is between i and j, and H_{L_j} > H_j, so the condition fails.)
    
    # So for a fixed j, the valid i's are i \in [L_j, j-1].
    # But we must also ensure i >= 1.
    # So for each j from 2 to N, the condition is satisfied for i = L_j, L_j + 1, ..., j-1.
    # Note: L_j is the index of the nearest element to the left larger than H_j.
    
    # Let's use a monotonic stack to find L_j for all j.
    L = [0] * N
    stack = [] # stores indices
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1] + 1 # 1-based index
        else:
            L[j] = 1 # 1-based index
        stack.append(j)
    
    # Now we have for each j, the range of i's: [L[j], j]. 
    # Wait, the range is i < j, so i can be L[j], L[j]+1, ..., j-1.
    # Let's use 0-indexing for internal logic:
    # For each j (0 to N-1), the condition is satisfied for i such that:
    # L_idx[j] <= i < j, where L_idx[j] is the index of the first element to the left > H[j].
    # If no such element, L_idx[j] = 0.
    # But if L_idx[j] is the index of a taller building, then i cannot be smaller than L_idx[j]
    # because then H[L_idx[j]] would be between i and j and would be taller than H[j].
    # Actually, if i = L_idx[j], the buildings between i and j are indices L_idx[j]+1 ... j-1.
    # All these are <= H[j] by definition of L_idx.
    # So for a fixed j, the valid i's are L_idx[j], L_idx[j]+1, ..., j-1.
    # Wait, if L_idx[j] is the index of a building taller than H[j], then i cannot be L_idx[j] - 1,
    # because then building L_idx[j] is between i and j.
    # So i must be >= L_idx[j].
    # Let's trace Sample 1: 2 1 4 3 5
    # j=0: H=2. L_idx=0. (No i < 0)
    # j=1: H=1. L_idx=0 (H[0]=2 > 1). i range: [0, 0]. i=0 satisfies.
    # j=2: H=4. L_idx=0. i range: [0, 1]. i=0, 1 satisfy.
    # j=3: H=3. L_idx=2 (H[2]=4 > 3). i range: [2, 2]. i=2 satisfies.
    # j=4: H=5. L_idx=0. i range: [0, 3]. i=0, 1, 2, 3 satisfy.
    
    # Let's count for each i:
    # i=0: j=1, 2, 4 (3)
    # i=1: j=2, 4 (2)
    # i=2: j=3, 4 (2)
    # i=3: j=4 (1)
    # i=4: (0)
    # Result: 3 2 2 1 0. Matches Sample 1!
    
    # Implementation:
    # For each j, we have a range [L_idx[j], j-1].
    # We want to count for each i how many ranges contain it.
    # This is a standard range update problem.
    # We can use a difference array.
    
    diff = [0] * (N + 1)
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        
        l_idx = stack[-1] + 1 if stack else 0
        # The condition is satisfied for i in [l_idx, j-1]
        if j > 0:
            # We need to be careful: the condition is "no building taller than Building j 
            # between i and j".
            # If i = l_idx, the buildings between are l_idx+1 ... j-1.
            # If l_idx is the index of the building taller than H[j],
            # then buildings between l_idx and j are indeed all <= H[j].
            # If l_idx is 0 (no building taller), then buildings between 0 and j 
            # are 1 ... j-1. These are all <= H[j].
            # So i can be as small as l_idx.
            
            # Correct logic:
            # Let L_idx[j] be the index of the first building to the left of j such that H[L_idx[j]] > H[j].
            # If no such building exists, L_idx[j] = -1.
            # The buildings between i and j are those with indices k: i < k < j.
            # We need H[k] <= H[j] for all such k.
            # This is true if and only if there is no k in (i, j) such that H[k] > H[j].
            # This means the first building to the left of j that is taller than H[j] 
            # must be at an index <= i.
            # So i >= L_idx[j].
            # Also we need i < j.
            # So i is in [max(0, L_idx[j]), j-1].
            
            # Wait, if L_idx[j] is the index of the taller building, 
            # then if i = L_idx[j], the range (i, j) is (L_idx[j], j).
            # The building at L_idx[j] is NOT between i and j.
            # So i can be L_idx[j].
            # If i = L_idx[j] - 1, then the building at L_idx[j] IS between i and j.
            # Since H[L_idx[j]] > H[j], this i is invalid.
            # So the minimum i is L_idx[j].
            # But we must also ensure i >= 0.
            
            # Let's re-calculate L_idx using 0-indexing:
            # stack stores indices.
            # for j in range(N):
            #     while stack and H[stack[-1]] < H[j]: stack.pop()
            #     l_idx = stack[-1] if stack else -1
            #     # Valid i: l_idx <= i < j
            #     # However, if l_idx == -1, the smallest i is 0.
            #     # If l_idx >= 0, the smallest i is l_idx.
            #     # Let's check Sample 1 again: 2 1 4 3 5
            #     # j=0: H=2. l_idx=-1. i in [0, -1] -> empty
            #     # j=1: H=1. l_idx=0. i in [0, 0] -> i=0
            #     # j=2: H=4. l_idx=-1. i in [0, 1] -> i=0, 1
            #     # j=3: H=3. l_idx=2. i in [2, 2] -> i=2
            #     # j=4: H=5. l_idx=-1. i in [0, 3] -> i=0, 1, 2, 3
            #     # i=0: j=1, 2, 4 (3)
            #     # i=1: j=2, 4 (2)
            #     # i=2: j=3, 4 (2)
            #     # i=3: j=4 (1)
            #     # i=4: 0
            #     # Result: 3 2 2 1 0. Perfect.
            
            # Using the logic:
            # l_idx = stack[-1] if stack else 0
            # if l_idx == 0 and not stack:
            #     # no building to the left is taller
            #     start = 0
            # else:
            #     start = l_idx
            # However, simpler:
            # l_idx = stack[-1] if stack else 0
            # If stack is empty, L_idx is effectively -1, but we can't have i < 0.
            # So for j, i is in [max(0, l_idx), j-1].
            # Wait, if stack is not empty, L_idx is stack[-1].
            # If stack is empty, L_idx is -1.
            # The range is [max(0, L_idx), j-1].
            
            # Let's refine the loop:
            pass

    # Resetting and implementing the refined logic
    diff = [0] * (N + 1)
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        
        l_idx = stack[-1] if stack else -1
        start = max(0, l_idx)
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
        stack.append(j)
    
    # Compute prefix sums of diff to get counts for each i
    results = []
    current = 0
    for i in range(N):
        current += diff[i]
        results.append(current)
    
    print(*(results))

if