import sys

def solve():
    # Increase recursion depth for deep stacks if necessary, 
    # though we use an iterative approach here.
    sys.setrecursionlimit(300000)
    
    # Read N
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of j > i 
    # such that H[k] <= H[j] for all i < k < j.
    #
    # Let's rephrase: j satisfies the condition if H[j] is a "prefix maximum" 
    # of the sequence H[i+1], H[i+2], ..., H[N].
    # However, the condition is specifically "between i and j", 
    # which means indices k where i < k < j.
    #
    # For a fixed i, j satisfies the condition if:
    # For all k such that i < k < j, H[k] <= H[j].
    #
    # This is equivalent to saying that H[j] must be greater than or equal to 
    # the maximum height of all buildings between i and j.
    #
    # Let's look at the problem from the perspective of j.
    # For a fixed j, which i < j satisfy the condition?
    # i must be such that max(H[i+1]...H[j-1]) <= H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = 0 (using 1-based indexing).
    # Then for any i such that L[j] <= i < j, the condition is satisfied.
    # Why? Because if i >= L[j], then all buildings between i and j are to the right of L[j],
    # and since L[j] was the first building taller than H[j], all buildings 
    # between L[j] and j are shorter than H[j].
    #
    # So for each j, it contributes to the count c_i for all i in [L[j], j-1].
    # The number of such i is j - L[j].
    #
    # We can use a monotonic stack to find L[j] for all j in O(N).
    
    L = [0] * N
    stack = [] # stores indices
    
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            # L[j] is the index of the building taller than H[j]
            # Using 0-based indexing for the array, but the logic 
            # requires the range [L[j]+1, j] in 1-based.
            # In 0-based: i can be L[j] up to j-1.
            # Example: H = [2, 1, 4], j=2 (H=4). Stack is empty. L[2]= -1.
            # i can be 0, 1.
            L[j] = stack[-1] + 1
        else:
            L[j] = 0
        stack.append(j)
        
    # Now we have L[j] for each j.
    # We need to find for each i, how many j > i satisfy L[j] <= i < j.
    # This is a range update problem: for each j, add 1 to range [L[j], j-1].
    # We can use a difference array (prefix sums) to solve this.
    
    diff = [0] * (N + 1)
    for j in range(N):
        # range is [L[j], j-1]
        start = L[j]
        end = j - 1
        if start <= end:
            diff[start] += 1
            diff[end + 1] -= 1
            
    # Calculate prefix sums to get c_i
    results = []
    current = 0
    for i in range(N):
        current += diff[i]
        results.append(current)
        
    print(*(results))

if __name__ == "__main__":
    solve()