import sys
from functools import reduce

# Increase recursion depth for deep building lines
sys.setrecursionlimit(300000)

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "right-side" visible building from i.
    # This is equivalent to saying Building j is part of the 
    # monotonic increasing sequence of heights starting from index i+1.
    
    # To solve this for all i efficiently, we can process from right to left.
    # For a building i, the buildings j that satisfy the condition are:
    # 1. Building i+1 (always satisfies if it exists)
    # 2. Any building j that was visible from i+1 and is taller than H[i+1].
    # Actually, a simpler observation: j satisfies the condition if 
    # H[j] > max(H[i+1]...H[j-1]).
    
    # Let's use a recursive function to simulate the process of counting
    # visible buildings using a jump-pointer logic (similar to a Segment Tree 
    # or Sparse Table idea, but simpler).
    # If H[i+1] > H[i], then i+1 is visible, and any building visible from i+1
    # is also visible from i.
    # If H[i+1] < H[i], then i+1 is visible, but we only care about buildings
    # taller than H[i+1] that are visible from i+1.
    
    # However, the condition is: no building between i and j is taller than H[j].
    # This means j is visible from i if H[j] is a prefix maximum of the 
    # sequence H[i+1], H[i+2]... H[N].
    
    # Let f(i) be the number of prefix maximums in H[i+1...N].
    # If i == N, f(N) = 0.
    # If i < N:
    # The first prefix maximum is always H[i+1].
    # The subsequent prefix maximums are the prefix maximums of H[i+2...N]
    # that are greater than H[i+1].
    
    # To implement this without loops, we use a memoized recursion.
    # next_taller[i] = the index of the first building j > i such that H[j] > H[i].
    
    # Step 1: Compute next_taller array using a stack-based approach via reduce.
    # We process indices in reverse.
    def get_next_taller(h_list):
        def step(state, i):
            stack, next_taller = state
            # Remove elements from stack that are smaller than current height
            # Since we can't use while loops, we use a helper recursive function.
            def pop_smaller(s):
                if not s or h_list[s[-1]] > h_list[i]:
                    return s
                return pop_smaller(s[:-1])
            
            new_stack = pop_smaller(stack)
            res = new_stack[-1] if new_stack else N
            return (new_stack + [i], next_taller + [res])
        
        # Process from N-1 down to 0
        final_state = reduce(step, range(N-1, -1, -1), ([], []))
        # The next_taller list was built by appending, so it's in reverse order (0 to N-1)
        # Wait, the reduce was range(N-1, -1, -1), so the first element added was for N-1.
        # So the resulting next_taller list is [res_{N-1}, res_{N-2}, ..., res_0].
        # We need it as [res_0, ..., res_{N-1}].
        return final_state[1][::-1]

    next_taller = get_next_taller(H)

    # Step 2: Compute counts using memoized recursion.
    # count(i) = 1 + count(next_taller[i]) if i < N else 0.
    # Note: the condition is about buildings j > i. 
    # For a fixed i, the buildings j satisfying the condition are:
    # j_1 = i + 1
    # j_2 = next_taller[j_1]
    # j_3 = next_taller[j_2] ... and so on.
    
    memo = {}
    def get_count(i):
        if i >= N:
            return 0
        if i in memo:
            return memo[i]
        # The first visible building is always i+1 (if it exists)
        # Then the next is the first building taller than H[i+1], and so on.
        # But wait, the condition is: no building between i and j is taller than H[j].
        # This means H[j] must be a prefix maximum of H[i+1...N].
        # The prefix maximums are:
        # 1. H[i+1]
        # 2. The first building after i+1 that is taller than H[i+1]
        # 3. The first building after that which is taller than the previous, etc.
        
        # So for index i, the count is 1 + get_count_from_height(i + 1)
        # where get_count_from_height(k) is the number of prefix maximums in H[k...N].
        # Let g(k) be the number of prefix maximums in H[k...N].
        # g(k) = 1 + g(next_taller[k]) if k < N else 0.
        
        # The answer for i is g(i+1).
        return 0 # This function is a placeholder for the logic below

    # Correcting the logic:
    # Let g(k) be the number of prefix maximums in the range [k, N-1].
    # g(k) = 1 + g(next_taller[k]) if k < N else 0.
    
    # To avoid recursion depth issues and loops, we can use reduce to build the g array.
    # Since g(k) depends on g(next_taller[k]) and next_taller[k] > k,
    # we compute g from N-1 down to 0.
    
    def compute_g(h_list, nt_list):
        def step(g_map, i):
            val = 1 + g_map.get(nt_list[i], 0) if i < N else 0
            return {**g_map, i: val}
        
        # Using a dictionary and reduce is slow for 2*10^5. 
        # Let's use a list and a loop-replacement.
        # Since we can't use loops, we use a recursive function with a list.
        pass

    # Final attempt at a loop-free implementation using a list and recursion for g.
    g = [0] * (N + 1)
    def fill_g(i):
        if i < 0:
            return
        g[i] = 1 + g[next_taller[i]] if i < N else 0
        fill_g(i - 1)

    # To avoid recursion depth for fill_g, we can use a trick with map/list comprehension
    # But g[i] depends on g[next_taller[i]], and next_taller[i] > i.
    # This is a perfect candidate for a right-to-left pass.
    # Since we can't use for-loops, we use a recursive function to simulate the loop
    # or use a technique to populate the list.
    
    # Actually, the most reliable way to do this without loops or deep recursion
    # is to use a custom reduce that updates a mutable list.
    def build_g(acc, i):
        acc[i] = 1 + acc[next_taller[i]] if i < N else 0
        return acc

    # We need to process i from N-1 down to 0.
    # We use a list for acc to allow mutation.
    g_final = reduce(build_g, range(N - 1, -1, -1), [0] * (N + 1))
    
    # The answer for i is g(i+1).
    results = [g_final[i+1] for i in range(N)]
    print(*(results))

if __name__ == "__main__":
    solve()