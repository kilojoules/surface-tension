import sys

def solve():
    # Increase recursion depth just in case, although not needed for this iterative approach
    sys.setrecursionlimit(300000)
    
    # Read N from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The problem asks for the number of j > i such that no building between i and j 
    # is taller than building j.
    # This means for a fixed i, we are looking for the number of "right-side visible" 
    # buildings starting from i.
    # Specifically, j satisfies the condition if H[j] > max(H[k]) for all i < k < j.
    # Note: The condition "no building taller than Building j between i and j"
    # implies that Building j must be the tallest among all buildings in the range (i, j].
    # Wait, a simpler interpretation:
    # For a fixed i, we are counting j > i such that H[j] > max(H[i+1], ..., H[j-1]).
    # This is equivalent to saying that as we iterate from j = i+1 to N,
    # we count j if H[j] is a new maximum encountered so far in the range [i+1, N].
    
    # However, N is up to 2*10^5, so an O(N^2) approach will be too slow.
    # We need a more efficient way.
    
    # Let's re-evaluate the condition:
    # For a fixed i, j satisfies the condition if H[j] > max_{i < k < j} H[k].
    # This is exactly the number of elements that would be kept in a "monotonic increasing stack"
    # if we processed elements from i+1 to N.
    
    # Let's look at the problem from a different perspective:
    # Building j is counted for building i if for all k such that i < k < j, H[k] < H[j].
    # This means j is the first building to the right of k (for all i < k < j) that is taller than H[k].
    # Actually, the condition "no building taller than H[j] between i and j" means
    # H[j] must be greater than the maximum of all buildings in the open interval (i, j).
    
    # Let's use a Monotonic Stack.
    # For a fixed j, for which i < j is the condition satisfied?
    # The condition is: max(H[i+1]...H[j-1]) < H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing for heights).
    # Then for any i such that L[j] <= i < j, the condition is satisfied.
    # Because for any i in [L[j], j-1], the range (i, j) contains no building taller than H[j].
    # Wait, if i = L[j], the range is (L[j], j). The tallest building in there is < H[j] by definition of L[j].
    # If i < L[j], the range (i, j) contains building L[j], and H[L[j]] > H[j], so the condition is failed.
    # So for a fixed j, the values of i that satisfy the condition are i = L[j], L[j]+1, ..., j-1.
    # That is (j-1) - L[j] + 1 = j - L[j] values of i.
    
    # Let's refine this:
    # For each j from 2 to N:
    # Find L[j] = max { k < j | H[k] > H[j] }. If no such k, L[j] = 0.
    # Then for all i in {L[j], L[j]+1, ..., j-1}, building j satisfies the condition.
    # Note: The problem asks for each i, the count of j.
    # So we can use a difference array or a Fenwick tree to add 1 to the range [L[j], j-1].
    
    # Let's trace Sample 1: H = [2, 1, 4, 3, 5]
    # j=1: H[1]=2. (No j > i, so not applicable for counting j)
    # j=2: H[2]=1. L[2]=1 (H[1]=2 > 1). i range: [1, 1]. i=1 gets +1.
    # j=3: H[3]=4. L[3]=0 (None > 4). i range: [0, 2]. Wait, i is 1-indexed.
    # Let's use 0-indexing for H: H = [2, 1, 4, 3, 5]
    # j=0: H[0]=2.
    # j=1: H[1]=1. L[1]=0 (H[0]=2 > 1). i in [0, 0]. i=0: count=1.
    # j=2: H[2]=4. L[2]=-1 (None > 4). i in [-1, 1]. But i must be >= 0. i in [0, 1]. i=0: 2, i=1: 1.
    # j=3: H[3]=3. L[3]=2 (H[2]=4 > 3). i in [2, 2]. i=2: count=1.
    # j=4: H[4]=5. L[4]=-1 (None > 5). i in [0, 3]. i=0: 3, i=1: 2, i=2: 2, i=3: 1.
    # Total for i: i=0: 3, i=1: 2, i=2: 2, i=3: 1, i=4: 0.
    # Matches sample!
    
    L = [-1] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = -1
        stack.append(j)
        
    # Now we need to count for each i, how many j > i satisfy L[j] <= i < j.
    # For a fixed j, the range of i is [L[j], j-1].
    # Note: if L[j] is -1, the range is [0, j-1].
    
    # We can use a difference array to calculate this.
    diff = [0] * (N + 1)
    for j in range(1, N):
        start = max(0, L[j])
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
            
    # Compute prefix sums of diff array
    results = []
    current = 0
    for i in range(N):
        current += diff[i]
        results.append(current)
        
    print(*(results))

if __name__ == "__main__":
    solve()