import sys
from functools import reduce

def solve():
    # Read input data
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
    # This means for all k where i < k < j, H[k] <= H[j].
    # This is equivalent to saying that Building j is visible from Building i
    # if we only care about buildings to the right of i.
    # Actually, the condition is: j satisfies the condition if max(H[i+1]...H[j-1]) <= H[j].
    # This is exactly the definition of a "right-to-left" monotonic stack problem
    # if we process from N down to 1.
    
    # For a fixed i, we want to count j > i such that for all k in (i, j), H[k] < H[j].
    # Note: H_i are distinct.
    # This is equivalent to counting how many elements to the right of i are 
    # "upper bounds" of the sequence starting at i+1.
    # Specifically, j satisfies the condition if H[j] > max(H[i+1], ..., H[j-1]).
    # The first element H[i+1] always satisfies this (since the set of k is empty).
    # Then the next j is the first index > i+1 such that H[j] > H[i+1], and so on.
    
    # This is a classic problem that can be solved by observing that the 
    # buildings satisfying the condition for i are exactly the elements of the
    # monotonic increasing stack when processing the suffix H[i+1:] from left to right.
    # But we need this for all i.
    
    # Let's use the property: the answer for i is the size of the monotonic stack
    # built from H[i+1...N].
    # We can solve this by processing the array from right to left.
    # We maintain a monotonic decreasing stack of values from the right.
    # Wait, the standard "visible buildings" problem is:
    # For a fixed i, j is visible if H[j] > max(H[i+1]...H[j-1]).
    # This means we are looking for the number of left-to-right maxima of the suffix H[i+1:].
    
    # Let f(i) be the number of left-to-right maxima of H[i+1...N].
    # If H[i+1] is the maximum of the suffix, f(i) = 1.
    # Otherwise, H[i+1] is the first maximum, and the remaining are the 
    # left-to-right maxima of the suffix starting from the first index k > i+1
    # such that H[k] > H[i+1].
    
    # Let next_greater[i] be the index of the first element to the right of i that is > H[i].
    # Then c_i = 1 + c_{next_greater[i+1]} (if next_greater[i+1] exists).
    
    # 1. Find next_greater for all indices using a stack.
    # 2. Compute c_i using the recurrence.
    
    # To avoid recursion depth issues, we use a loop to compute c_i.
    # Since we need c_i for i=1...N, and c_i depends on next_greater[i+1],
    # we can compute this using a list and a loop from N down to 1.
    
    # Using a list comprehension to find next_greater is tricky, 
    # so we use a standard stack approach.
    
    def get_next_greater(arr):
        n = len(arr)
        res = [n] * n
        stack = []
        for idx in range(n - 1, -1, -1):
            while stack and arr[stack[-1]] < arr[idx]:
                stack.pop()
            if stack:
                res[idx] = stack[-1]
            stack.append(idx)
        return res

    ng = get_next_greater(H)
    
    # c[i] is the answer for Building i.
    # The condition is about j > i. The first candidate is j = i + 1.
    # The number of such j is 1 + (number of such j for index i+1 starting from ng[i+1]).
    # Let dp[k] be the number of left-to-right maxima in the suffix starting at k.
    # dp[k] = 1 + dp[ng[k]] if k < N else 0.
    
    # We can use reduce to build the dp array from right to left.
    # However, since we need to access dp[ng[k]], a simple reduce is hard.
    # But we can use a list and a loop.
    
    # Since I must provide the code in a single block and avoid explicit for-loops 
    # for the DP if possible (though loops are allowed, I'll keep it clean),
    # I will use a list and a loop to calculate the DP.
    
    # Note: the problem asks for i = 1 to N.
    # For i = N, c_N = 0.
    # For i < N, the first j is i+1. The others are the left-to-right maxima of H[i+2...N]
    # that are greater than H[i+1]. This is exactly dp[ng[i+1]] if we define
    # dp[k] as the number of left-to-right maxima of the suffix starting at k.
    
    # Correct DP:
    # Let dp[k] = number of left-to-right maxima of H[k...N-1].
    # dp[k] = 1 + dp[ng[k]] (with dp[N] = 0).
    # Then for building i (0-indexed), the answer is dp[i+1].
    
    # To compute dp without a loop, we can't easily because of the jumps.
    # But the constraints allow a loop.
    
    # Using a list and a loop to compute dp:
    # We can't use a loop to mutate a list in a functional way, 
    # but we can use a list and a loop to fill it.
    
    # Wait, I can use a list and a loop to compute the DP values.
    # But I need to be careful about the "no loop" preference in some environments.
    # Actually, the prompt says "complete Python program", loops are fine.
    
    # Let's implement the DP.
    
    # Using a list and a loop:
    # dp = [0] * (N + 1)
    # for k in range(N - 1, -1, -1):
    #     dp[k] = 1 + dp[ng[k]]
    # result = dp[1:]
    
    # To do this without a loop, we can use the fact that the dependency 
    # is a forest structure and we are summing depths. 
    # But a loop is the most straightforward way.
    
    # Let's use a list and a loop.
    
    # Re-checking the logic:
    # For i=1, H=[2, 1, 4, 3, 5]
    # j=2: H[2]=1. Max(empty)= -inf. 1 > -inf (Yes)
    # j=3: H[3]=4. Max(H[2])= 1. 4 > 1 (Yes)
    # j=4: H[4]=3. Max(H[2], H[3])= 4. 3 > 4 (No)
    # j=5: H[5]=5. Max(H[2], H[3], H[4])= 4. 5 > 4 (Yes)
    # Total for i=1: 3.
    # This matches the logic: left-to-right maxima of H[i+1...N].
    
    # Implementation of the DP using a loop:
    # Since I cannot use a loop to build the list in a 'functional' way,
    # I will use a list and a loop.
    
    # Actually, I can use a technique to avoid the loop by using the 
    # property that we are summing values along a path in a forest.
    # But for N=2e5, a simple loop is O(N) and perfectly fine.
    
    # Final logic assembly:
    # 1. ng = get_next_greater(H)
    # 2. dp = [0] * (N + 1)
    # 3. for k in range(N-1, -1, -1): dp[k] = 1 + dp[ng[k]]
    # 4. print(*dp[1:])
    
    # Let's wrap this in a function and call it.
    pass

# Since the logic above was in a 'solve' function that didn't execute, 
# I will rewrite it into a final executable form.

def main():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # Find next greater element index for each position
    # ng[i] is the smallest j > i such that H[j] > H[i]
    ng = [N] * N
    stack = []
    for idx in range(N - 1, -1, -1):
        while stack and H[stack[-1]] < H[idx]:
            stack.pop()
        if stack:
            ng[idx] = stack[-1]
        stack.append(idx)

    # dp[k] = number of left-to-right maxima in suffix H[k...N-1]
    # dp[k] = 1 + dp[ng[k]]
    dp = [0] * (N + 1)
    for k in range(N - 1, -1, -1):
        dp[k] = 1 + dp[ng[k]]

    # The answer for building i (1-indexed) is the number of 
    # left-to-right maxima of the suffix starting at i+1 (0-indexed).
    # So for i=1, we need dp[1]. For i=N, we need dp[N].
    print(*(dp[1:]))

if __name__ == "__main__":
    main()