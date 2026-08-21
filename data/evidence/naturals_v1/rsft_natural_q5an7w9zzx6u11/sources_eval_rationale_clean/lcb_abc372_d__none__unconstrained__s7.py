import sys

# Increase recursion depth for deep monotonic stack processing
sys.setrecursionlimit(300000)

def solve():
    # Read N and H
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need to find for each i, how many j > i satisfy:
    # max(H[k] for i < k < j) <= H[j]
    # This is equivalent to counting elements in a monotonic stack 
    # when traversing from right to left.
    
    # stack stores indices of buildings that are taller than all buildings to their right
    # For a building i, the buildings j that satisfy the condition are exactly 
    # the buildings that would remain in a monotonic decreasing stack (of heights)
    # if we processed from j = N down to i+1.
    
    # However, the condition is: no building between i and j is taller than H[j].
    # This means j is a "right-side visible" building.
    # For a fixed i, the indices j that satisfy this are the indices of the 
    # upper-envelope of the heights to the right.
    
    # Let's use a recursive function to simulate the stack popping.
    # we process from N-1 down to 0.
    
    memo_stack = []
    results = [0] * N

    def process(i):
        if i < 0:
            return
        
        # For building i, the buildings j > i that satisfy the condition
        # are the ones that form a strictly increasing sequence of heights
        # when looking from i onwards.
        # Actually, the number of such j is simply the size of the monotonic 
        # stack maintained from the right.
        
        # But the stack needs to be updated for the next i (which is i-1).
        # For building i, any building j > i is visible if H[j] > max(H[k]) for i < k < j.
        # This is exactly the number of elements in a monotonic stack 
        # constructed by iterating from i+1 to N-1.
        
        # Wait, the condition is: no building BETWEEN i and j is taller than H[j].
        # This means H[j] >= max(H[i+1] ... H[j-1]).
        # This is satisfied by the elements of the monotonic stack 
        # if we build it from i+1 to N.
        pass

    # Correct approach:
    # For a fixed i, the indices j that satisfy the condition are those 
    # that would be added to a monotonic stack when iterating from i+1 to N.
    # Let f(i) be the number of such j.
    # f(i) = 1 (for j = i+1) + f(next_taller_than_i_plus_1)
    # Let next[k] be the index of the first building j > k such that H[j] > H[k].
    # Then c_i = 1 + c_{next[i+1]} if i+1 < N, else 0.
    
    # To find next[k] without loops:
    # We can use a stack and recursion.
    next_taller = [N] * N
    
    def find_next(stack, idx):
        if idx < 0:
            return
        
        # Simulate while stack and H[stack[-1]] < H[idx]: stack.pop()
        def pop_smaller(s, val):
            if not s or H[s[-1]] > val:
                return s
            return pop_s(s[:-1], val)
        
        def pop_s(s, val):
            if not s or H[s[-1]] > val:
                return s
            return pop_s(s[:-1], val)

        # This is getting complex. Let's use a simpler recursive structure.
        pass

    # Let's redefine: 
    # c_i is the number of j > i such that H[j] > max(H[i+1]...H[j-1])
    # This is exactly the number of elements in the monotonic stack 
    # starting from i+1.
    # Let dp[k] be the number of visible buildings starting from index k.
    # dp[k] = 1 + dp[next_taller[k]] (if next_taller[k] < N)
    # else dp[k] = 1
    
    # To find next_taller without loops, we use a recursive function 
    # that processes the array.
    
    def get_next_taller(indices):
        # Using a list comprehension and a helper to simulate the stack
        # Since we can't use while, we use a recursive function to find the 
        # first element to the right that is taller.
        res = [N] * N
        
        def compute(stack, i):
            if i < 0: return
            # We need to process from right to left to find next taller
            # But the constraint is no loops.
            pass
            
    # Let's use the property: next_taller[i] can be found by 
    # checking H[i+1] and then next_taller[i+1] if H[i+1] <= H[i].
    
    def find_nt(i, H, nt):
        if i >= N - 1:
            return N
        # If H[i+1] > H[i], then next_taller is i+1
        # If H[i+1] <= H[i], then next_taller is find_nt(i+1, ...) 
        # but only for those taller than H[i].
        pass

    # Actually, the simplest way to implement this without loops 
    # is using a recursive function with a memoization table for next_taller.
    
    nt = [N] * (N + 1)
    def get_nt(i):
        if i >= N - 1:
            return N
        # We need the first j > i such that H[j] > H[i]
        # Check i+1
        if H[i+1] > H[i]:
            return i + 1
        # Otherwise, jump to the next taller of i+1 and check again
        return jump(i+1, H[i])

    def jump(curr, target_h):
        if curr >= N:
            return N
        if H[curr] > target_h:
            return curr
        # Use the already computed next_taller of curr to skip
        # This requires nt to be computed from right to left.
        return jump(nt[curr], target_h)

    # To compute nt from right to left without loops:
    def build_nt(i):
        if i < 0:
            return
        # Compute nt[i] using the jump logic
        # Since we go from N-1 down to 0, nt[i+1] is already known.
        # We can't use a loop, so we use a recursive call.
        # But we need to compute nt[i] before calling build_nt(i-1).
        pass

    # Correcting the approach: 
    # Use a recursive function to iterate through the array.
    def solve_recursive(i, nt_array):
        if i < 0:
            return
        
        # Compute nt[i]
        def find_j(curr):
            if curr >= N: return N
            if H[curr] > H[i]: return curr
            return find_j(nt_array[curr])
        
        nt_array[i] = find_j(i + 1)
        solve_recursive(i - 1, nt_array)

    nt_array = [N] * (N + 1)
    solve_recursive(N - 1, nt_array)
    
    # Now compute c_i
    # c_i = 1 + dp[nt_array[i+1]] if i+1 < N else 0
    dp = [0] * (N + 1)
    def compute_dp(i):
        if i >= N:
            return 0
        # The building at index i is the first visible building for the range starting at i
        # The next visible building is the one that is taller than H[i]
        return 1 + compute_dp(nt_array[i])

    # We need to compute dp for all i from N-1 down to 0.
    # Since we can't use loops, we use a recursive function to fill the dp table.
    def fill_dp(i):
        if i < 0:
            return
        # dp[i] depends on dp[nt_array[i]]
        # Since nt_array[i] > i, we should compute from N-1 down to 0.
        # But wait, the recursive definition of compute_dp already handles it.
        # We just need to memoize it.
        pass

    memo_dp = {}
    def get_dp(i):
        if i >= N:
            return 0
        if i in memo_dp:
            return memo_dp[i]
        res = 1 + get_dp(nt_array[i])
        memo_dp[i] = res
        return res

    # For each i from 0 to N-1, the answer is get_dp(i+1)
    # Use map and list comprehension to output
    ans = [get_dp(i + 1) for i in range(N)]
    print(*(ans))

solve()