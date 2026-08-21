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
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] < H[j] for all i < k < j.
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum when looking at the range [i+1, j].
    # More simply: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means the sequence of heights H[j] that satisfy this for a fixed i
    # are the elements of the sequence H[i+1...N] that are strictly greater 
    # than all preceding elements in that suffix.
    
    # However, the problem asks for this for every i.
    # Let's rephrase: j satisfies the condition for i if for all k such that i < k < j, H[k] < H[j].
    # This is exactly the definition of elements that would remain in a 
    # monotonic stack if we processed the array from j down to i+1.
    # Actually, a simpler way to think about it:
    # For a fixed i, we are counting j > i such that H[j] > max_{i < k < j} H[k].
    # This is equivalent to counting how many indices j > i are "visible" 
    # from position i looking right, where a building j is visible if 
    # it is taller than all buildings between i and j.
    
    # Note: The condition does NOT depend on H[i]. It only depends on buildings 
    # strictly between i and j.
    # So for a fixed i, we are looking at the sequence H[i+1], H[i+2], ..., H[N].
    # We want to count j such that H[j] > max(H[i+1], ..., H[j-1]).
    # This is simply the number of prefix maximums of the suffix H[i+1:].
    
    # To do this efficiently for all i:
    # Let f(i) be the number of prefix maximums of H[i+1...N].
    # If we know the answer for i+1, how does it change for i?
    # The sequence for i is H[i+1], H[i+2], ..., H[N].
    # The sequence for i+1 is H[i+2], ..., H[N].
    # The first element H[i+1] is always a prefix maximum.
    # Then we count how many j > i+1 are prefix maximums of H[i+2...N] 
    # AND are taller than H[i+1].
    
    # Let next_greater[i] be the index of the first building j > i such that H[j] > H[i].
    # The number of visible buildings from i is:
    # 1 (for building i+1) + (number of visible buildings from i+1 that are taller than H[i+1]).
    # Wait, the condition is about buildings BETWEEN i and j.
    # For a fixed i, j=i+1 always satisfies it (no buildings between).
    # For j > i+1, j satisfies it if H[j] > max(H[i+1...j-1]).
    # This means H[j] must be a prefix maximum of the suffix starting at i+1.
    # The prefix maximums of H[i+1...N] are:
    # 1. Index i+1
    # 2. The first index j1 > i+1 such that H[j1] > H[i+1]
    # 3. The first index j2 > j1 such that H[j2] > H[j1]
    # ... and so on.
    
    # This is a jump-pointer problem. 
    # Let next_greater[i] = min {j > i | H[j] > H[i]} or N if no such j exists.
    # The number of such j for a given i is the length of the chain:
    # (i+1) -> next_greater[i+1] -> next_greater[next_greater[i+1]] -> ...
    
    # 1. Compute next_greater array using a stack
    next_greater = [N] * N
    stack = []
    for idx in range(N):
        while stack and H[stack[-1]] < H[idx]:
            next_greater[stack.pop()] = idx
        stack.append(idx)
        
    # 2. Compute the count using dynamic programming
    # dp[i] = 1 + dp[next_greater[i]] if i < N else 0
    # We process from N-1 down to 0.
    dp = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        nxt = next_greater[i]
        dp[i] = 1 + (dp[nxt] if nxt < N else 0)
        
    # The answer for i is the number of prefix maximums of H[i+1...N].
    # This is exactly dp[i+1] (if i+1 < N).
    results = [dp[i+1] if i+1 < N else 0 for i in range(N)]
    
    print(*(results))

if __name__ == "__main__":
    solve()