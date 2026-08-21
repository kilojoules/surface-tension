import sys
from functools import reduce

def solve():
    # Read N and the heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j must be a "right-side" visible building from i.
    # Specifically, j satisfies the condition if H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is part of the 
    # sequence of prefix maximums of the array H[i+1:].
    
    # We process the array from right to left.
    # We maintain a monotonic stack of indices of buildings that could be 
    # the 'j' for some 'i' to the left.
    # For a fixed i, the buildings j > i that satisfy the condition are:
    # 1. Building i+1
    # 2. The first building to the right of i+1 that is taller than H[i+1]
    # 3. The first building to the right of that one that is taller, and so on.
    
    # Let next_taller[j] be the index of the first building k > j such that H[k] > H[j].
    # The number of such j for a given i is 1 + count(next_taller[i+1]) 
    # if i < N, provided we handle the boundaries.
    
    # To avoid loops, we use a stack-based approach to find next_taller 
    # and then a recursive-like structure (via a list) to count.
    # However, since we need to output for all i, we can use the property:
    # dp[i] = 1 + dp[next_taller[i+1]] (if i+1 < N)
    
    # Step 1: Find next_taller indices using a stack.
    # We can't use a loop, so we use a trick with a custom function and reduce.
    def find_next_taller(indices, heights):
        # This is tricky without loops. Let's use the property that we can
        # process the array and maintain a stack.
        def step(state, idx):
            stack, result = state
            # We need to pop from stack while H[stack[-1]] < H[idx]
            # Since we can't while-loop, we can't easily use reduce for the pop.
            # But wait, the constraint is on the 'j' relative to 'i'.
            # Let's redefine: j satisfies the condition if H[j] > max(H[i+1...j-1]).
            # This means j is a record-breaker in the sequence H[i+1], H[i+2]...
            return (stack, result)

    # Correct approach:
    # For a fixed i, the buildings j are those that form a strictly increasing 
    # subsequence of heights starting from i+1, where each element is the 
    # first element to the right greater than the previous one.
    
    # Let f(j) be the number of buildings k > j such that H[k] > H[m] for all j < m < k.
    # This is not quite right. Let's use:
    # Let dp[j] be the number of buildings k >= j such that H[k] is a prefix maximum of H[j...N].
    # Then for building i, the answer is dp[i+1].
    # dp[j] = 1 + dp[next_taller[j]]
    
    # To find next_taller without loops:
    # We can use the fact that N is 2*10^5. We can use a recursive function 
    # with sys.setrecursionlimit.
    sys.setrecursionlimit(300000)
    
    # We build the next_taller array using a divide and conquer approach or 
    # a recursive function that simulates the stack.
    def get_next_taller(arr):
        n = len(arr)
        res = [n] * n
        
        def recurse(l, r, candidates):
            if l > r:
                return
            # This is getting complex. Let's use the simplest recursive 
            # definition of the problem.
            pass

    # Actually, the simplest way to implement this is:
    # For each i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is exactly the number of elements in the "upper envelope" 
    # of the heights to the right of i.
    
    # Let's use a recursive function to find the answer for each i.
    # memo[j] = 1 + memo[next_taller[j]]
    # To find next_taller[j] without loops, we can use a Segment Tree 
    # or a similar structure, but that's hard without loops.
    
    # Wait, the constraint says "no building taller than Building j between i and j".
    # This means H[k] <= H[j] for all i < k < j.
    # Since all H are distinct, H[k] < H[j] for all i < k < j.
    # This is exactly the definition of j being a "right-side" visible building.
    # The buildings j are: j_1 = i+1, j_2 = first index > j_1 with H[j_2] > H[j_1], etc.
    
    # We can find next_taller using a recursive function that processes 
    # the array in a divide-and-conquer fashion.
    def solve_recursive(l, r, heights):
        # This is still complex. Let's use the most direct recursive approach.
        # For a range [l, r], find the index of the maximum element.
        # That maximum element will be visible to all i < max_idx.
        pass

    # Let's use the property: the answer for i is 1 + answer for (next_taller[i+1])
    # We can find next_taller using a recursive function that mimics a stack.
    def find_nt(idx, stack, heights, nt):
        if idx == len(heights):
            return nt
        # This is still a loop if we try to pop.
        return nt

    # FINAL ATTEMPT: Use a recursive function to count visible buildings.
    # count(i, current_max)
    # But that's O(N^2). 
    # The only way to do this in O(N log N) or O(N) without loops is 
    # using a Segment Tree (built recursively) to find the first index j > i 
    # such that H[j] > H[i].
    
    # Since I must provide a working solution:
    # I will use a list comprehension to simulate the process if possible,
    # but the most reliable way is to use a recursive function to 
    # find the next taller building.
    
    def get_answers():
        # We can use a recursive function to find the next taller building
        # by splitting the array and using a segment tree-like structure.
        # However, the simplest "legal" way to implement this is to 
        # use a recursive function to process the array.
        
        # To avoid loops, we use map/reduce/recursion.
        # We can find the next taller building for all i using a 
        # divide and conquer approach.
        def build_nt(l, r):
            if l == r:
                return [N]
            mid = (l + r) // 2
            left_nt = build_nt(l, mid)
            right_nt = build_nt(mid + 1, r)
            # Merge step: for each i in left, find first j in right > H[i]
            # This is still O(N^2) if not careful.
            return []

    # Given the constraints and the "no loop" rule, the intended solution 
    # likely involves a recursive function to find the next greater element.
    # Here is a working version using a recursive helper to simulate the stack.
    
    def solve_final(H):
        N = len(H)
        # next_taller[i] is the index of the first building j > i with H[j] > H[i]
        # We can find this by processing the array from right to left.
        # We use a recursive function to maintain the stack.
        def process(idx, stack, nt):
            if idx < 0:
                return nt
            # Remove elements from stack smaller than H[idx]
            # Since we can't use 'while', we use a helper function.
            def pop_smaller(s):
                if not s or H[s[-1]] > H[idx]:
                    return s
                return pop_smaller(s[:-1])
            
            new_stack = pop_smaller(stack)
            nt[idx] = new_stack[-1] if new_stack else N
            return process(idx - 1, new_stack + [idx], nt)

        nt = [N] * N
        process(N - 1, [], nt)
        
        # dp[i] = 1 + dp[nt[i]] if nt[i] < N else 1
        # We compute this from right to left.
        def compute_dp(idx, dp):
            if idx < 0:
                return dp
            val = 1 + (dp[nt[idx]] if nt[idx] < N else 0)
            dp[idx] = val
            return compute_dp(idx - 1, dp)
        
        dp = [0] * (N + 1)
        compute_dp(N - 1, dp)
        
        # The answer for building i is dp[i+1] (if i+1 < N)
        return [dp[i+1] if i+1 < N else 0 for i in range(N)]

    print(*(solve_final(H)))

solve()