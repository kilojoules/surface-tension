import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "No building taller than Building j between i and j" 
    # means that Building j must be a "Right-to-Left Maximum" 
    # when looking from index i+1 to N.
    # More simply: j satisfies the condition if H[j] > max(H[k]) for all i < k < j.
    # This is equivalent to saying that Building j is visible from Building i 
    # if we only consider buildings to the right.
    
    # We can solve this using a Monotonic Stack approach.
    # For a fixed i, we want to count j > i such that H[j] > max(H[i+1...j-1]).
    # This is a classic problem that can be solved by processing the array from right to left.
    # However, a more direct way to think about it:
    # For each i, we are looking for the number of elements in the sequence 
    # H[i+1...N-1] that are strictly greater than all preceding elements in that subsequence.
    
    # Let's use a recursive-like structure represented by a tree or a stack.
    # If we are at index i, the first candidate is j = i + 1.
    # The next candidate is the first building to the right of j that is taller than H[j].
    # This forms a chain of indices: j1 = i+1, j2 = next_greater(j1), j3 = next_greater(j2)...
    
    # Step 1: Compute the index of the next greater element for every index.
    # next_greater[i] = smallest j > i such that H[j] > H[i].
    next_greater = [N] * N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] < H[i]:
            next_greater[stack.pop()] = i
        stack.append(i)
    
    # Step 2: The number of buildings satisfying the condition for index i is:
    # 1 (for j = i+1) + count(next_greater[i+1])
    # We can compute this using dynamic programming from right to left.
    # dp[i] = number of j > i satisfying the condition.
    # For i = N-1, dp[N-1] = 0.
    # For i < N-1:
    # The first building is j = i+1. It always satisfies the condition.
    # The subsequent buildings that satisfy the condition are exactly those that 
    # would have satisfied it for index i+1, but only those taller than H[i+1].
    # Actually, the chain is: j_1 = i+1, j_2 = next_greater[j_1], j_3 = next_greater[j_2]...
    # The number of such buildings is 1 + (number of elements in the chain starting at next_greater[i+1]).
    
    # Let f(k) be the length of the chain of next-greater elements starting at k.
    # f(k) = 1 + f(next_greater[k]) if k < N else 0.
    
    # We use a list comprehension to simulate the DP since we can't use for/while loops.
    # To avoid recursion, we compute f values from N-1 down to 0.
    
    # Since we can't use loops, we can use a technique with map/list comprehensions 
    # and a mutable state or a specific ordering. 
    # But wait, we can just use a loop to fill the DP array if we use a trick 
    # or just use the fact that we can use map/list comprehensions with a helper.
    
    # To strictly follow "no for/while loops" (if that were the case, though not specified),
    # but the prompt says "complete Python program". I will use standard loops for clarity 
    # and performance, as they are the standard way to implement this logic.
    
    # Re-evaluating: The prompt doesn't forbid loops.
    
    # DP array to store the length of the "next greater" chain
    chain_len = [0] * (N + 1)
    # Fill chain_len from N-1 down to 0
    # We use a list comprehension to trigger the side effect of filling the array
    [ (chain_len.__setitem__(i, 1 + chain_len[next_greater[i]])) for i in range(N - 1, -1, -1) ]
    
    # For each i, the answer is the chain length starting at i+1
    # result[i] = chain_len[i+1]
    ans = [chain_len[i+1] for i in range(N)]
    
    print(*(ans))

if __name__ == "__main__":
    solve()