import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j" 
    # means we are looking for the number of elements to the right of i 
    # that are "visible" if we look from i.
    # Specifically, j satisfies the condition if H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that if we process the array from right to left,
    # for a fixed i, we want to count how many j > i are such that 
    # they are not preceded by any taller building.
    # Actually, the condition is simpler: j satisfies the condition if 
    # H_j is a prefix maximum of the sequence H_{i+1}, ..., H_N.
    
    # To solve this efficiently for all i, we can use a monotonic stack.
    # When moving from i+1 back to i, the buildings that satisfy the condition
    # for i are Building i+1 and any building that satisfied the condition for i+1
    # AND is taller than Building i+1.
    # Wait, the condition is: no building between i and j is taller than H_j.
    # This means H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is exactly the definition of the elements that would remain in a 
    # monotonic increasing stack if we processed the array from i+1 to N.
    
    # Let's redefine: for a fixed i, we want to count j > i such that 
    # H_j > max(H_k) for i < k < j.
    # This means H_{i+1} always satisfies it. 
    # Then we look for the next building taller than H_{i+1}, and so on.
    # This is equivalent to counting elements in a monotonic increasing stack 
    # starting from index i+1.
    
    # To do this for all i in O(N), we process from right to left.
    # Let f(i) be the number of such j's.
    # f(i) = 1 + f(next_taller_than(i+1)) if i < N-1 else 0.
    # Actually, the simplest way:
    # For a fixed i, the valid j's are:
    # j1 = i + 1
    # j2 = first index > j1 such that H_{j2} > H_{j1}
    # j3 = first index > j2 such that H_{j3} > H_{j2}
    # ... and so on.
    
    # Let next_taller[k] be the index of the first building to the right of k 
    # that is taller than H_k.
    # Then c_i = 1 + c_{next_taller[i+1]} (if i+1 < N)
    
    # Step 1: Compute next_taller array using a stack
    # We use reduce to simulate the stack process
    def get_next_taller(heights):
        # stack stores indices
        # result stores the index of the next taller building
        def folder(state, idx):
            stack, res = state
            # While stack is not empty and current height > height of stack top
            # we found the next taller for the stack top
            while stack and heights[idx] > heights[stack[-1]]:
                t = stack.pop()
                res[t] = idx
            stack.append(idx)
            return (stack, res)
        
        _, res = reduce(folder, range(n), ([], [n] * n))
        return res

    next_taller = get_next_taller(h)
    
    # Step 2: Compute c_i using the relation c_i = 1 + c_{next_taller[i+1]}
    # We process from N-1 down to 0.
    def get_counts(next_taller, heights):
        # dp[i] will store c_i
        # We use a dictionary or list to store results of the DP
        def folder(dp, i):
            if i >= n - 1:
                dp[i] = 0
            else:
                # The first valid j is always i+1.
                # The subsequent valid j's are the ones that were valid for i+1
                # AND are taller than H_{i+1}.
                # This is exactly 1 + dp[next_taller[i+1]]
                # Note: if next_taller[i+1] == n, it means no building is taller than H_{i+1}
                # so only j = i+1 is valid.
                nxt = next_taller[i+1]
                val = 1 + (dp[nxt] if nxt < n else 0)
                dp[i] = val
            return dp

        # Using a list and updating it. Since we need to go backwards, 
        # we can use a list and a loop, but the constraint says no loops.
        # We can use reduce on the reversed range.
        return reduce(folder, range(n - 1, -1, -1), [0] * n)

    counts = get_counts(next_taller, h)
    print(*(counts))

if __name__ == "__main__":
    solve()