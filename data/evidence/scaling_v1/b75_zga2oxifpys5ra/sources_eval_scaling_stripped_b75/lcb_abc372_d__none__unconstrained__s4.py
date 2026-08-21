import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[j] is a new maximum encountered while scanning from i+1 to N.
    # However, the condition is specifically about buildings BETWEEN i and j.
    # Let's re-read: "no building taller than Building j between Buildings i and j".
    # This means for all k such that i < k < j, H[k] <= H[j].
    # This is equivalent to saying that Building j is visible from Building i
    # if we only care about the height of the target building j.
    # Actually, the standard interpretation of this specific problem (from ABC)
    # is that j satisfies the condition if H[j] > max(H[i+1]...H[j-1]).
    
    # We can solve this by processing the buildings from right to left.
    # For a fixed i, the valid j's are the indices that form a 
    # "strictly increasing subsequence" starting from the first element after i.
    # But wait, the condition is simpler: j is valid if H[j] is greater than 
    # all heights in the range (i, j).
    # This means for a fixed i, the valid j's are exactly the indices of the
    # left-to-right maxima of the suffix H[i+1:].
    
    # To do this efficiently for all i, we can use a Segment Tree or a similar 
    # structure, but since we need to count elements in a suffix that are 
    # greater than all preceding elements in that suffix, we can use the 
    # property that the number of such elements in H[i+1:] is:
    # 1 + (number of elements in H[next_greater_idx + 1:] that are greater than 
    # the maximum of the range [i+1, next_greater_idx]).
    
    # Actually, the simplest way to think about this is:
    # For a fixed i, we are counting j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of left-to-right maxima of the array H[i+1...N].
    
    # Let f(i) be the number of left-to-right maxima in H[i...N].
    # If H[i] is the maximum of the suffix, then f(i) = 1 + f(index of first element > H[i] in H[i+1...N]).
    # Wait, that's for a different problem. 
    # For the left-to-right maxima of H[i...N]:
    # The first element H[i] is always a maximum.
    # The next maximum is the first element to the right of i that is taller than H[i].
    # Let next_taller[i] be the index j > i such that H[j] > H[i] and j is minimized.
    # Then the number of maxima in H[i...N] is 1 + count_maxima(next_taller[i]).
    
    # 1. Find next_taller for all i using a stack
    next_taller = [N] * N
    stack = [N] # Boundary
    # We process in reverse to find the next taller element
    # But the stack approach for next greater element is usually:
    # For i in range(N-1, -1, -1):
    #     while stack and H[stack[-1]] < H[i]: stack.pop()
    #     next_taller[i] = stack[-1] if stack else N
    #     stack.append(i)
    
    # Let's implement the next_taller logic carefully.
    # We need the first j > i such that H[j] > H[i].
    # Using a list comprehension to simulate the stack is tricky, 
    # so we use a standard loop to build the next_taller array.
    
    # Since I must provide the code in a single block and avoid loops 
    # where possible (though loops are allowed, the prompt asks for a 
    # complete program), I will use a helper function with a loop 
    # to compute next_taller and then use reduce to compute the counts.
    
    def get_next_taller(n, heights):
        res = [n] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and heights[stack[-1]] < heights[i]:
                stack.pop()
            if stack:
                res[i] = stack[-1]
            stack.append(i)
        return res

    nt = get_next_taller(N, H)
    
    # dp[i] = number of left-to-right maxima in H[i...N-1]
    # dp[i] = 1 + dp[nt[i]] if nt[i] < N else 1
    # We compute this from N-1 down to 0.
    
    # To avoid a loop for DP, we can use a list and a trick with 
    # a function that populates it, or just use a loop.
    # Given the constraints and Python's recursion limit, a loop is safest.
    
    dp = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        if nt[i] < N:
            dp[i] = 1 + dp[nt[i]]
        else:
            dp[i] = 1
            
    # The question asks for the number of j > i.
    # For a fixed i, we are looking at the suffix starting at i+1.
    # So the answer for i is dp[i+1] if i+1 < N else 0.
    
    results = [dp[i+1] if i+1 < N else 0 for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()