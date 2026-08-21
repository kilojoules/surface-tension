import sys

def solve():
    # Read N from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H_k <= H_j for all i < k < j.
    # This is equivalent to saying that Building j is a "visible" building
    # when looking to the right from Building i, but specifically 
    # based on the heights of the buildings encountered.
    # Actually, the condition is: j satisfies the condition if 
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # Note: For j = i + 1, the set of buildings between is empty, 
    # so the condition is vacuously true.
    
    # Let's rephrase: for a fixed i, we want to count j > i such that
    # H_j is a new maximum in the sequence H_{i+1}, H_{i+2}, ..., H_N.
    # Wait, that's not quite right. The condition is "no building taller than H_j".
    # If H_{i+1} = 10 and H_{i+2} = 5, then for j=i+2, building i+1 (height 10)
    # is taller than building j (height 5), so j=i+2 fails.
    # If H_{i+1} = 5 and H_{i+2} = 10, then for j=i+2, building i+1 (height 5)
    # is not taller than building j (height 10), so j=i+2 succeeds.
    
    # Correct interpretation:
    # For a fixed i, j satisfies the condition if H_j >= max(H_{i+1}, ..., H_{j-1}).
    # Since all H are distinct, this is H_j > max(H_{i+1}, ..., H_{j--1}).
    # This means j is a "prefix maximum" of the sequence starting from index i+1.
    
    # However, we need to do this for all i. 
    # Let's look at it from the perspective of j.
    # Building j is counted for building i if for all k such that i < k < j, H_k < H_j.
    # This means i must be greater than the index of the first building to the left of j 
    # that is taller than H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H_{L[j]} > H_j.
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for a fixed j, the condition is satisfied for all i such that L[j] <= i < j.
    # Wait, the condition is "between i and j". 
    # If i = L[j], the buildings between i and j are {L[j]+1, ..., j-1}.
    # By definition of L[j], all these buildings are shorter than H_j.
    # So i = L[j] satisfies the condition.
    # If i < L[j], then building L[j] is between i and j, and H_{L[j]} > H_j,
    # so the condition is violated.
    # Thus, for a fixed j, the valid i's are L[j], L[j]+1, ..., j-1.
    # The number of such i's is j - L[j].
    
    # But we need the count for each i.
    # For each j from 2 to N:
    #   It contributes 1 to the count c_i for i in range(L[j], j).
    # This is a range update problem. We can use a difference array to solve it.
    
    # To find L[j] for all j, we use a monotonic stack.
    l = [0] * (n + 1)
    stack = []
    for j in range(1, n + 1):
        height = h[j-1]
        while stack and h[stack[-1]-1] < height:
            stack.pop()
        if stack:
            l[j] = stack[-1]
        else:
            l[j] = 0
        stack.append(j)
        
    # Difference array for range updates
    # We want to add 1 to c_i for i from L[j] to j-1.
    # Note: i must be at least 1.
    diff = [0] * (n + 2)
    for j in range(1, n + 1):
        # The condition is for i < j.
        # The range of i is max(1, L[j]) <= i <= j-1.
        start = max(1, l[j])
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
            
    # Compute prefix sums to get the actual counts
    results = []
    current = 0
    for i in range(1, n + 1):
        current += diff[i]
        results.append(str(current))
        
    sys.stdout.write(" ".join(results) + "\n")

if __name__ == "__main__":
    solve()