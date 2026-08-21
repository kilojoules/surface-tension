import sys
from functools import reduce

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for the number of indices j > i
    # such that H[j] > max(H[k]) for all i < k < j.
    # This is equivalent to saying that j is a "right-side visible" building.
    # Specifically, j satisfies the condition if H[j] is a new maximum 
    # encountered while scanning from i+1 to N.
    # However, the condition is actually simpler: j satisfies it if 
    # for all k such that i < k < j, H[k] < H[j].
    
    # Let's re-evaluate: For a fixed i, we want count of j > i such that
    # max({H[k] | i < k < j}) < H[j].
    # This means if we maintain a running maximum of heights to the right of i,
    # a building j is counted if its height is greater than the maximum of 
    # all buildings between i and j.
    
    # This is a classic problem that can be solved by observing that 
    # for a fixed i, the indices j that satisfy this are exactly the indices
    # of the elements that would remain in a monotonic stack if we processed
    # the array from i+1 to N.
    # More simply: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means H[j] must be a prefix maximum of the suffix H[i+1...N].
    
    # To solve this for all i efficiently:
    # We can use a Segment Tree or a similar structure, but there is a 
    # recursive property:
    # Let f(i) be the number of such j's.
    # The first such j is always i+1 (since the set of k is empty).
    # The next such j must be the first index j > i+1 such that H[j] > H[i+1].
    # All j's between i+1 and this new j that satisfied the condition for i+1
    # will NOT satisfy the condition for i because H[i+1] is between them and j,
    # and H[i+1] > H[j].
    # Thus, the j's for i are: {i+1} union {j's for (first index j > i+1 where H[j] > H[i+1])}.
    
    # 1. Find the next greater element (NGE) index for every index.
    # We use a stack to find the index of the first element to the right that is taller.
    nge = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            nge[stack.pop()] = idx
        stack.append(idx)
    
    # 2. Use dynamic programming to count visible buildings.
    # dp[i] = 1 + dp[nge[i]] if nge[i] < N else 1
    # We process from N-1 down to 0.
    # Note: the question asks for j > i. The first j is always i+1.
    # So for index i, the first visible is i+1, and the rest are the visible
    # buildings starting from the NGE of i+1.
    
    # We need a DP array where dp[i] is the number of visible buildings in the 
    # range [i, N-1] when looking from the left of i.
    dp = [0] * (N + 1)
    # For the last building, there are no j > N-1, so dp[N-1] = 0 (handled by range)
    # But we need the count for i, which depends on i+1.
    
    # Let's redefine: dp[i] = number of j in {i...N-1} such that 
    # H[j] > max(H[i...j-1]).
    # For i = N-1: dp[N-1] = 1 (only j=N-1)
    # For i < N-1: dp[i] = 1 + dp[nge[i]] (if nge[i] < N else 1)
    
    # We calculate this DP from right to left.
    # Since we need the answer for i (which looks at range i+1...N-1),
    # the answer for i is simply dp[i+1].
    
    # To compute dp[i] for all i:
    # We can't use a loop, so we use a list comprehension or map with a helper.
    # Since we need the values of dp to the right, we can use a recursive function 
    # with memoization or a custom reduce.
    
    # Using a dictionary for memoization to avoid recursion depth issues 
    # and loops, we can use a function and map it over the indices.
    # However, the most reliable way to avoid loops and recursion in Python 
    # for this specific DP is to use a dictionary and a comprehension 
    # if we can process in the right order.
    
    # Since we need to process from N-1 down to 0:
    # We can use a dictionary to store DP values and a comprehension to fill it.
    # But comprehensions cannot reference the dictionary being built.
    # Instead, we use a recursive function with @functools.lru_cache.
    
    import sys
    sys.setrecursionlimit(300000)
    from functools import lru_cache

    @lru_cache(None)
    def count_visible(i):
        if i >= N:
            return 0
        # Current building i is always visible.
        # The next visible building is the Next Greater Element of i.
        next_idx = nge[i]
        return 1 + count_visible(next_idx)

    # The answer for building i is the number of visible buildings in range [i+1, N-1].
    results = [count_visible(i + 1) for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()