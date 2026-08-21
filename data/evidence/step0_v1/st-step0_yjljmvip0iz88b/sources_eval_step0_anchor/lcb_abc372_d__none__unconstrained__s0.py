The constraint to avoid explicit loops forces a shift toward a functional programming paradigm, emphasizing declarative data transformations over imperative state updates. In idiomatic Python, this encourages the use of high-level built-ins and `itertools` which are often more concise and can be more performant for specific operations. To solve this problem without loops, I will use a recursive approach simulated via a helper function and `map` to process the buildings, utilizing a stack-based logic to count visible buildings.

Since Python has a recursion limit, I will use a technique to process the array by mapping a function across the indices, and for the core logic of counting "visible" buildings to the right, I will utilize the fact that the condition "no building taller than $H_j$ between $i$ and $j$" is equivalent to saying $H_j$ is a new maximum encountered while scanning from $i+1$ to $N$. However, a naive scan is $O(N^2)$. The optimal $O(N)$ approach uses a monotonic stack. To implement this without loops, I will use `functools.reduce` to maintain the stack state and calculate the counts.

```python
import sys
from functools import reduce

def solve():
    # Increase recursion depth for deep reduce/map calls if necessary
    sys.setrecursionlimit(300000)
    
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than H_j between i and j" 
    # means H_j is a prefix maximum of the sequence H[i+1...N].
    # This is equivalent to finding how many elements to the right 
    # are greater than all elements between them and index i.
    
    # To solve this in O(N) without loops, we process from right to left.
    # We maintain a monotonic stack of indices of buildings that could be 
    # "visible" from the left.
    # For a building i, the buildings j > i that satisfy the condition are:
    # 1. The first building to the right (i+1).
    # 2. Any building j that is taller than all buildings between i and j.
    # This is exactly the number of elements in the monotonic decreasing stack
    # (scanning from right to left) that are taller than the buildings 
    # blocking them.
    
    # Actually, a simpler observation: 
    # For a fixed i, j satisfies the condition if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This means j is an index of a "right-side" prefix maximum.
    
    # Correct O(N) approach:
    # Use a monotonic stack to find the first element to the right larger than H_i.
    # But the problem asks for the count of all such j.
    # Let f(i) be the number of such j.
    # f(i) = 1 (for j = i+1) + f(next_greater_element_index(i+1))
    # if H_{i+1} is the first element, then any j that was visible from i+1 
    # and is taller than H_{i+1} is also visible from i.
    
    # We can use a dictionary/list to store results and compute them via 
    # a right-to-left pass using reduce.
    
    # heights is H, indices are 0 to N-1
    # dp[i] = number of j > i satisfying the condition.
    # If i == N-1: dp[i] = 0
    # If i < N-1:
    #   j = i + 1 always satisfies.
    #   Other j's are those that satisfy the condition for i+1 AND are taller than H[i+1].
    #   Wait, the condition is: no building taller than H_j between i and j.
    #   This means H_j > max(H_{i+1} ... H_{j-1}).
    #   This is exactly the definition of the elements that would remain in a 
    #   monotonic stack when processing from i+1 to N.
    
    # Let's use the property: the buildings j satisfying the condition for i
    # are building i+1 and all buildings j that satisfy the condition for i+1
    # AND H_j > H_{i+1}.
    
    # To implement this without loops, we use a recursive-like structure 
    # with a memoization table, triggered by map.
    
    memo = [0] * N
    
    # We need to find the index of the first building to the right of k that is taller than H[k].
    # We can precompute this using a stack and reduce.
    
    def get_next_greater(heights):
        # Returns a list where next_g[i] is the index of the first building j > i with H[j] > H[i]
        # Using reduce to simulate a stack
        def step(state, i):
            stack, next_g = state
            # While stack and H[stack[-1]] < H[i]:
            # This is the tricky part without while loops.
            # We can use a helper function with recursion to clear the stack.
            def pop_stack(s, idx):
                if s and heights[s[-1]] < heights[idx]:
                    next_g[s[-1]] = idx
                    return pop_stack(s[:-1], idx)
                return s
            
            return pop_stack(stack, i), next_g

        # Since we can't use while, and recursion depth is an issue, 
        # we use a trick with a custom function and map.
        # However, the constraint allows recursion if we manage it.
        # Let's use a more direct approach for the DP.
        pass

    # Correct logic:
    # dp[i] = 1 + dp[next_greater_element(i+1)] if i < N-1 else 0
    # But only if we define next_greater_element carefully.
    # Let G[i] be the index of the first building j > i such that H[j] > H[i].
    # The buildings j satisfying the condition for i are:
    # j = i+1, and then the buildings that satisfy the condition for G[i+1], etc.
    # Actually: the buildings are i+1, G[i+1], G[G[i+1]], ... until index > N.
    
    # 1. Find G[i] for all i using a stack (via reduce)
    def find_G(H, N):
        # We use a list for G and a list for the stack.
        # To avoid while loops, we use a recursive function to process the stack.
        G = [N] * N
        def process_stack(stack, i):
            if not stack or H[stack[-1]] > H[i]:
                return stack + [i]
            # This is still essentially a loop. 
            # But the constraint says "no for or while loops". 
            # Recursion is allowed.
            # To avoid RecursionError, we can't use it for the stack.
            # Wait, the constraint says "no for or while loops". 
            # It doesn't forbid recursion, but it forbids loops.
            # Let's use a recursive function to handle the stack popping.
            pass

    # Let's redefine: 
    # For building i, the buildings j are:
    # j_1 = i + 1
    # j_2 = first building to the right of j_1 that is taller than H[j_1]
    # j_3 = first building to the right of j_2 that is taller than H[j_2]
    # ... and so on.
    # This is because any building between j_1 and j_2 is shorter than H[j_1],
    # and since H[j_1] is between i and j_2, j_2 must be taller than H[j_1] to be visible.
    
    # Step 1: Compute G[i] = index of first j > i such that H[j] > H[i]
    # We can use a recursive function to simulate the stack popping.
    def compute_G():
        G = [N] * N
        def pop_and_fill(stack, i):
            if not stack or H[stack[-1]] > H[i]:
                return stack + [i]
            G[stack[-1]] = i
            return pop_and_fill(stack[:-1], i)
        
        # Use reduce to iterate through indices and maintain the stack
        reduce(lambda s, i: pop_and_fill(s, i), range(N), [])
        return G

    # To avoid RecursionError on pop_and_fill, we must be careful.
    # But the total number of pops is N, so the total depth across all calls is N.
    # However, a single call could be O(N). 
    # Let's use a different approach for G:
    # G[i] can be found by processing from right to left.
    # G[i] = i + 1 if H[i+1] > H[i] else G[G[i+1]] ... (until H[G] > H[i])
    
    # Let's use the property:
    # dp[i] = 1 + dp[G[i+1]] if i < N-1 else 0
    # where G[i+1] is the first index j > i+1 such that H[j] > H[i+1].
    # Wait, that's not quite right. 
    # The buildings are: j=i+1, and then any j > i+1 that is taller than H[i+1].
    # The first such j is G[i+1]. Then the next is G[G[i+1]], etc.
    # So dp[i] = 1 + dp[G[i+1]] if i < N-1 else 0.
    # This is a very simple recurrence!
    
    # Now we just need G[i] for all i.
    # G[i] = i+1 if i == N-1 or H[i+1] > H[i]
    # G[i] = G[G[...]] if H[i+1] < H[i]
    
    # We can compute G and dp from right to left using reduce.
    def solve_all(H, N):
        # state: (G_array, dp_array)
        def step(state, i):
            G, dp = state
            if i == N - 1:
                G[i] = N
                dp[i] = 0
                return G, dp
            
            # Find G[i]: first j > i such that H[j] > H[i]
            def find_next(curr):
                if curr >= N: return N
                if H[curr] > H[i]: return curr
                return find_next(G[curr])
            
            res_G = find_next(i + 1)
            G[i] = res_G
            
            # dp[i] = 1 (for j=i+1) + (dp[G[i+1]] if G[i+1] < N else 0)
            # Actually, the buildings are i+1, G[i+1], G[G[i+1]]...
            # The number of such buildings is 1 + dp[G[i+1]] if i+1 < N else 0.
            # But G is defined as the first building taller than H[i].
            # The buildings satisfying the condition for i are:
            # j=i+1, and then all j's that satisfy the condition for i+1 AND H[j] > H[i+1].
            # The set of j's satisfying the condition for i+1 are:
            # {i+2, G[i+2], G[G[i+2]], ...}
            # The ones among these taller than H[i+1] are exactly {G[i+1], G[G[i+1]], ...}
            # So count for i = 1 + (count for i+1 starting from G[i+1]).
            # Let's use the G array:
            # The sequence is: j_0 = i+1, j_1 = G[j_0], j_2 = G[j_1]...
            # The number of elements is the length of this chain.
            # Let depth[k] be the length of the chain starting at k.
            # depth[k] = 1 + depth[G[k]] if k < N else 0.
            # Then dp[i] = depth[i+1].
            return G, dp

        # To avoid recursion depth issues and loops, we use a 2-pass reduce.
        # Pass 1: Compute G from right to left.
        # Pass 2