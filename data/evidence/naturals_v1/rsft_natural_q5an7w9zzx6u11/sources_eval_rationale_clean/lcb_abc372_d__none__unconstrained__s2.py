import sys

# Increase recursion depth for deep DP chains and large N
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need to find the index of the first building to the right of i 
    # that is taller than H[i]. Let's call this next_greater[i].
    # To avoid loops, we use a recursive function to build the next_greater array.
    # However, a standard monotonic stack is iterative. 
    # We can simulate the monotonic stack using recursion.
    
    def get_next_greater(index, stack):
        if index == N:
            return [], []
        
        # While stack is not empty and current height > height of building at stack top
        if stack and H[index] > H[stack[-1]]:
            # The current index is the next greater element for the building at the top of the stack
            # We need to handle the stack popping. Since we can't use while, 
            # we use a helper function to "pop" and record.
            return process_stack(index, stack)
        else:
            # Current building is smaller or stack is empty, just push and move next
            res_indices, res_values = get_next_greater(index + 1, stack + [index])
            return res_indices, res_values

    # To avoid the overhead of list concatenation in recursion and the 
    # constraints of no loops, we use a different approach for next_greater:
    # We can use a recursive function that processes the array from right to left.
    
    # next_greater[i] stores the index of the first building j > i such that H[j] > H[i]
    # If no such building exists, we use N.
    next_greater = [N] * N
    
    # We use a recursive function to simulate the monotonic stack logic
    # by passing the current "candidate" indices.
    def find_ng(i, candidates):
        if i < 0:
            return
        
        # Remove candidates that are shorter than current H[i]
        # Since we can't use while, we filter the candidates list.
        # But filtering is O(N), making the whole thing O(N^2).
        # To keep it O(N), we must use the property that candidates are sorted by height.
        pass

    # Given the constraints and the "no loop" rule, the most idiomatic 
    # way to implement this is using a recursive function with a 
    # shared state (list) and using slices or map/filter.
    # However, the simplest O(N) "no-loop" approach for Next Greater Element 
    # is using a recursive function that mimics the stack.
    
    ng = [N] * N
    def build_ng(i, stack):
        if i == N:
            return
        
        # We need to pop from stack while H[i] > H[stack[-1]]
        # We use a helper function to handle the "while" logic via recursion
        def pop_stack(stk):
            if stk and H[i] > H[stk[-1]]:
                ng[stk[-1]] = i
                return pop_stack(stk[:-1])
            return stk
        
        new_stack = pop_stack(stack)
        build_ng(i + 1, new_stack + [i])

    build_ng(0, [])

    # Now we calculate c_i.
    # c_i = 1 + c_{next_greater[i]} if next_greater[i] < N else 0
    # But wait, the condition is: no building taller than H[j] between i and j.
    # This means j is a "visible" building if H[j] is a running maximum of H[i+1...j].
    # The number of such j is:
    # 1 (for the first building i+1) + 
    # (number of visible buildings starting from the first building taller than H[i+1])
    # Actually, the simplest relation is:
    # Let f(i) be the number of visible buildings to the right of i.
    # The first visible building is always j = i + 1.
    # The next visible building must be taller than H[i+1].
    # So it's 1 + f_starting_from(next_greater[i+1])
    
    # Let's redefine: dp[i] is the number of visible buildings in the range [i, N-1].
    # For the problem, we want c_i, which is the number of visible buildings in [i+1, N-1].
    # The first visible building is always i+1 (if i+1 < N).
    # The subsequent visible buildings are those that are visible starting from 
    # the first building to the right of i+1 that is taller than H[i+1].
    
    dp = [0] * (N + 1)
    def compute_dp(i):
        if i >= N:
            return 0
        # The building at index i is the first visible one.
        # The next visible one is the first building taller than H[i], 
        # and all buildings visible from that point onwards.
        # However, the "next greater" logic applies to the sequence.
        # Let's use the property: c_i = 1 + dp[next_greater[i+1]] 
        # where dp[k] is the count of visible buildings starting from index k.
        pass

    # Correct logic:
    # For a fixed i, the visible buildings are:
    # j1 = i + 1
    # j2 = next_greater[j1]
    # j3 = next_greater[j2] ... and so on.
    
    # We can compute this using DP from right to left.
    # dp[i] = 1 + dp[next_greater[i]] if i < N else 0
    
    # To avoid loops, we use a recursive function to fill the DP table.
    memo = [-1] * (N + 1)
    def get_count(i):
        if i >= N:
            return 0
        if memo[i] != -1:
            return memo[i]
        memo[i] = 1 + get_count(ng[i])
        return memo[i]

    # The answer for i is get_count(i + 1)
    # We use map to generate the results for all i from 0 to N-1.
    results = map(lambda i: get_count(i + 1), range(N))
    print(*(results))

solve()