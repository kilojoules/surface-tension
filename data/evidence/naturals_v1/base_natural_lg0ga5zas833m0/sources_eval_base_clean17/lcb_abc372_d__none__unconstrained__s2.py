```python
import sys

def solve():
    # Increase recursion depth for deep trees or recursions, 
    # though not strictly needed for this iterative stack approach.
    sys.setrecursionlimit(300000)
    
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # We need to find for each i, the number of j > i such that 
    # for all k (i < k < j), H_k <= H_j.
    # This is equivalent to saying that Building j is a "visible" 
    # building when looking right from Building i.
    # Specifically, Building j is counted if H_j is greater than 
    # all heights between i and j.
    
    # Let's analyze the condition: j is counted for i if 
    # max(H_{i+1}, ..., H_{j-1}) < H_j.
    # This means that for a fixed i, as we move j from i+1 to N, 
    # we only count j if H_j is a new maximum among the heights encountered 
    # since i.
    
    # However, the problem asks for this for all i. 
    # A naive O(N^2) approach will be too slow (N = 2*10^5).
    # We need a more efficient way.
    
    # Observation: 
    # For a fixed j, it is counted for i if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that there is no k such that i < k < j and H_k > H_j.
    # In other words, i must be greater than or equal to the index of the 
    # first building to the left of j that is taller than Building j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, it satisfies the condition for all i such that L[j] <= i < j.
    # Note: The condition "no building taller than Building j between i and j" 
    # means for all k in {i+1, ..., j-1}, H_k <= H_j.
    # Since all H are distinct, H_k < H_j.
    # The indices i that satisfy this are i = j-1, j-2, ..., L[j].
    # The number of such i is j - L[j].
    
    # However, the question asks for each i, how many j's.
    # This is a range update problem. For each j from 1 to N:
    # increment the count for all i in range [L[j], j-1].
    
    # L[j] can be found using a monotonic stack in O(N).
    l = [0] * n
    stack = []
    for j in range(n):
        while stack and h[stack[-1]] < h[j]:
            stack.pop()
        if stack:
            l[j] = stack[-1] + 1 # 1-based index of the taller building
        else:
            l[j] = 0 # No taller building to the left
        stack.append(j)
        
    # Now we have ranges [L[j], j-1] for each j.
    # We want to count how many such ranges cover each i.
    # We can use a difference array (prefix sum array) to solve this.
    diff = [0] * (n + 2)
    for j in range(n):
        # Building j (index j+1) contributes to i in [L[j]+1, j+1]
        # But the problem says i < j. So i is in [L[j]+1, j]
        # Note: L[j] is the index of the first building to the left taller than j.
        # The condition "no building taller than Building j between i and j" 
        # is satisfied for i if i is at or to the right of the nearest 
        # building taller than Building j.
        # Let's re-verify with Sample 1: 2 1 4 3 5
        # j=1 (H=2): L=0. i in [1, 0] -> None
        # j=2 (H=1): L=1 (H=2). i in [1, 1] -> i=1. (Between 1 and 2, no buildings)
        # j=3 (H=4): L=0. i in [1, 2] -> i=1, 2.
        # j=4 (H=3): L=3 (H=4). i in [3, 3] -> i=3.
        # j=5 (H=5): L=0. i in [1, 4] -> i=1, 2, 3, 4.
        
        # Let's use 0-based indexing for the implementation:
        # j is the index of the building (0 to N-1).
        # L[j] is the index of the first building to the left taller than h[j].
        # If no such building, L[j] = -1.
        # The condition is satisfied for i such that L[j] <= i < j.
        # (Wait, if L[j] is the index of the taller building, then for i = L[j], 
        # the buildings between i and j are {L[j]+1, ..., j-1}.
        # None of these are taller than h[j] because L[j] was the FIRST one.
        # So i can be as small as L[j].)
        
        # Correct range for i (0-indexed): [L[j], j-1]
        # But if L[j] = -1, the range is [0, j-1].
        
        start = max(0, l[j]) 
        # Wait, the logic above: L[j] was the index of the taller building.
        # Let's use the monotonic stack properly.
        pass

def solve_final():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # L[j] = index of the nearest building to the left of j that is taller than h[j]
    l = [-1] * n
    stack = []
    for j in range(n):
        while stack and h[stack[-1]] < h[j]:
            stack.pop()
        if stack:
            l[j] = stack[-1]
        stack.append(j)
        
    # For each j, the condition is satisfied for i in [L[j] + 1, j - 1]
    # Actually, if L[j] is the index of the first taller building, 
    # any i > L[j] will have no building taller than h[j] between i and j.
    # Example: H = [2, 1, 4, 3, 5]
    # j=0 (H=2): L=-1. i in [0, -1] -> None
    # j=1 (H=1): L=0. i in [1, 0] -> None (Wait, i < j, so i=0. Between 0 and 1, nothing. 0 is ok)
    # Let's re-evaluate:
    # i=0, j=1: buildings between are {}. Max is -inf. -inf < 1. OK.
    # i=0, j=2: buildings between are {1}. Max is H[1]=1. 1 < 4. OK.
    # i=0, j=3: buildings between are {1, 2}. Max is H[2]=4. 4 < 3. NO.
    # i=0, j=4: buildings between are {1, 2, 3}. Max is H[2]=4. 4 < 5. OK.
    # For j=1 (H=1), L=0. i can be 0.
    # For j=2 (H=4), L=-1. i can be 0, 1.
    # For j=3 (H=3), L=2. i can be 2.
    # For j=4 (H=5), L=-1. i can be 0, 1, 2, 3.
    
    # Correct logic: 
    # j satisfies the condition for i if max(H_{i+1}...H_{j-1}) < H_j.
    # This means no k in {i+1...j-1} has H_k > H_j.
    # This is true if i+1 > L[j], where L[j] is the index of the nearest taller building to the left.
    # So i > L[j] - 1, or i >= L[j].
    # Also we need i < j.
    # So i is in [L[j] + 1, j - 1]. 
    # Let's check: j=1, L=0. i in [1, 0]. Still wrong.
    # Let's re-read: "no building taller than Building j between Buildings i and j".
    # Between i and j are indices i+1, ..., j-1.
    # If i = j-1, the set is empty. Condition always true.
    # If i = j-2, the set is {j-1}. Condition: H_{j-1} < H_j.
    # If L[j] is the index of the first building to the left of j such that H_{L[j]} > H_j.
    # Then for any i such that L[j] <= i < j, the buildings between i and j are 
    # a subset of {L[j]+1, ..., j-1}.
    # Since L[j] was the FIRST building taller than H_j, all buildings to its right 
    # (up to j-1) are shorter than H_j.
    # Thus, any i in {L[j], L[j]+1, ..., j-1} works.
    # Wait, if i = L[j], the buildings between are {L[j]+1, ..., j-1}. All are < H_j. OK.
    # If i = L[j]-1, the buildings between are {L[j], ..., j-1}. H_{L[j]} > H_j. NOT OK.
    # So i must be in [L[j] + 1, j] if we use 1-based indexing for i and j.
    # In 0-based: i is in [L[j] + 1, j - 1]... no.
    # Let's use 0-based for indices: i, j in {0, ..., N-1}.
    # Condition: for all k: i < k < j, H_k < H_j.
    # This holds if i >= L[j], where L[j] is the index of the nearest building to the left 
    # such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = -1.
    # But we must have i < j.
    # So i is in {max(0, L[j] + 1), ..., j-1}.
    # Wait, if i = L[j], the buildings between are {L[j]+1, ..., j-1}. 
    # All these are < H_j by definition of L[j].
    # So i = L[j] also works!
    # Let's check: H = [2, 1, 4, 3, 5]
    # j=1 (H=1): L=0. i can be 0. (Between 0 and 1: empty). OK.
    # j=2 (H=4): L=-1. i can be 0, 1. (Between 0 and 2: {1}. H[1]=1 < 4. OK).
    # j=3 (H=3): L=2. i can be 2. (Between 2 and 3: empty). OK.
    # j=4 (H=5): L=-1. i can be 0, 1, 2, 3. (Between 0 and 4: {1,2,3}. Max is 4 < 5. OK).
    
    # So for each j, the range of i is [max(0, L[j] + 1), j - 1].
    # Let's