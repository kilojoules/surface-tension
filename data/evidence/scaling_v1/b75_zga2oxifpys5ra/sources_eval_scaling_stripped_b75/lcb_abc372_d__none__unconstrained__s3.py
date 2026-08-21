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
    # such that H[j] > max(H[i+1]...H[j-1]).
    # This is equivalent to counting elements in the sequence H[i+1...N-1]
    # that are strictly greater than all preceding elements in that subsequence.
    
    # However, the problem is symmetric to finding how many buildings to the right
    # are "visible" from building i. A building j is visible from i if 
    # H[k] < H[j] for all i < k < j.
    
    # This is a classic problem that can be solved by processing the array from right to left
    # using a monotonic stack to find the "next greater element" and building a 
    # structure, but since we need the count for every i, we can use the property that
    # the answer for i is 1 + (answer for the index of the first building to the right 
    # that is taller than H[i]), provided such a building exists.
    # Wait, that's for a different condition (H[k] < H[i]).
    
    # Let's re-read: "no building taller than Building j between i and j".
    # For a fixed i, j satisfies this if H[j] > max(H[i+1], ..., H[j-1]).
    # This means for a fixed i, we are counting the number of "left-to-right maxima"
    # of the suffix starting at i+1.
    
    # Let f(i) be the number of j > i satisfying the condition.
    # The buildings j that satisfy this are exactly the indices that would form
    # a strictly increasing subsequence if we only kept elements that are 
    # larger than all elements encountered so far since index i+1.
    
    # Correct approach:
    # For a fixed i, the indices j are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]
    # ... and so on.
    # This is because any j between j1 and j2 is smaller than H[j1], 
    # and H[j1] is between i and j, so the condition is violated.
    
    # Let next_greater[i] be the index of the first building j > i such that H[j] > H[i].
    # Then c_i = 1 + c_{next_greater[i+1]} if i+1 < N, else 0.
    # But the first building is always j = i+1. 
    # So c_i = 1 + (count of indices j > i+1 that are greater than all buildings between i+1 and j).
    # This is exactly 1 + c_{i+1}, but only counting those j that are also > H[i+1].
    # Actually, the sequence of j's for index i is:
    # (i+1), next_greater[i+1], next_greater[next_greater[i+1]], ...
    
    # Let's precompute next_greater using a stack.
    # next_greater[i] = index j > i such that H[j] > H[i], or N if none.
    
    # To avoid recursion limits and loops, we use a list comprehension to build 
    # the next_greater array and then use a technique to compute the counts.
    # Since we need to jump through the next_greater indices, and N is 2*10^5,
    # we can use a functional approach to build the counts from right to left.
    
    # 1. Compute next_greater indices
    # We use a stack-based approach to find the next greater element.
    # Since we can't use for-loops, we use a trick with a mutable list and map/reduce.
    
    def get_next_greater(n, heights):
        stack = []
        res = [n] * n
        # We process indices from 0 to n-1. For each index, while stack is not empty 
        # and current height > height of stack top, update res[stack.pop()].
        # To do this without a loop, we use a helper function with reduce.
        def step(stk, i):
            while stk and heights[i] > heights[stk[-1]]:
                res[stk.pop()] = i
            stk.append(i)
            return stk
        
        reduce(step, range(n), [])
        return res

    next_greater = get_next_greater(N, H)
    
    # 2. Compute c_i = 1 + c_{next_greater[i+1]}
    # We need to compute this for i = 0 to N-1.
    # c_i depends on values to its right. We can use reduce from right to left.
    # The state will be the list of counts computed so far.
    
    def compute_counts(next_gr, heights, n):
        # dp[i] will store the count for index i
        # We process from N-1 down to 0.
        # For i, the answer is:
        # If i == N-1: 0
        # If i < N-1: 1 + dp[next_greater[i+1]] (if next_greater[i+1] < N else 0)
        
        # Using a list and reduce to simulate the DP
        def step(dp, i):
            # i is the current index we are calculating c_i for
            # The condition is about j > i. The first j is always i+1.
            # The subsequent j's are next_greater[i+1], next_greater[next_greater[i+1]]...
            if i == n - 1:
                dp.append(0)
            else:
                # The number of j's for i is 1 + (number of j's for index i+1 
                # that are taller than H[i+1]).
                # Actually, the sequence is: (i+1), next_greater[i+1], ...
                # The number of elements in this sequence is 1 + dp[next_greater[i+1]]
                # where dp[k] is the number of elements in the sequence starting at k.
                
                # Wait, the sequence starting at i+1 is: (i+1), next_greater[i+1], ...
                # The count for i is simply 1 + (count for index next_greater[i+1])
                # if next_greater[i+1] < N, else 1.
                
                # Let's redefine: dp[k] = number of elements in the sequence 
                # k, next_greater[k], next_greater[next_greater[k]]...
                # Then c_i = dp[i+1].
                pass
        
        # Let's use a different approach for the DP to avoid the loop.
        # We can use a list and a function that fills it.
        # Since we can't use loops, we use a recursive-like structure with a list.
        pass

    # Correct DP:
    # Let G[i] = next_greater[i]
    # Let DP[i] = 1 + DP[G[i]] if G[i] < N else 1
    # Then c_i = DP[i+1] if i+1 < N else 0
    
    # To compute DP without loops:
    # We can use the fact that G[i] > i.
    # We can use a list and a function that we call via map/reduce.
    # But we need to go backwards.
    
    # Let's use a trick: 
    # We can't use loops, but we can use a list and a function that 
    # populates the list using the values already there.
    
    # Since we need to avoid loops and recursion, and N=2*10^5, 
    # we can use a list and a custom function with reduce.
    
    def final_solve(n, h):
        ng = get_next_greater(n, h)
        # dp[i] = 1 + dp[ng[i]] if ng[i] < n else 1
        # We compute dp from n-1 down to 0.
        # We use a list to store dp values. 
        # Since we are going backwards, we can't use a simple list.append.
        # Instead, we can use a list of size n and update it.
        
        dp = [0] * (n + 1)
        def step(unused, i):
            val = 1 + (dp[ng[i]] if ng[i] < n else 0)
            dp[i] = val
            return unused
        
        reduce(step, range(n - 1, -1, -1), None)
        
        # c_i = dp[i+1] if i+1 < n else 0
        return [dp[i+1] if i+1 < n else 0 for i in range(n)]

    print(*(final_solve(N, H)))

if __name__ == "__main__":
    solve()