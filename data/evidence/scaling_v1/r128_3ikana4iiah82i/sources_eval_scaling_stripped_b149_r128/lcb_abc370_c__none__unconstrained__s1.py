import sys
from itertools import product

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    M = len(diff_indices)

    # To minimize the array X lexicographically, we want to change characters 
    # at indices that make the resulting string as small as possible.
    # We should prioritize changing characters at the earliest possible indices 
    # to the target characters, BUT only if the target character is smaller 
    # than the current character. 
    # Actually, the problem is simpler: we must change M characters.
    # To make the sequence of strings X_1, X_2, ... X_M lexicographically smallest,
    # we want X_1 to be as small as possible, then X_2, and so on.
    
    # A character change at index i makes the string smaller if T[i] < S[i].
    # We should perform all "reducing" changes first, in order of index.
    # Then perform all "increasing" changes in reverse order of index.
    
    # Let's refine: 
    # We want to reach T from S in M steps. Each step changes one S[i] to T[i].
    # To make X_1 smallest, we want to change the first index i where T[i] < S[i].
    # If no such i exists, we must change the last index i where T[i] > S[i] 
    # to keep the string as small as possible for as long as possible.
    
    # Correct Strategy for Lexicographical Smallest X:
    # 1. Identify indices where T[i] < S[i] (Decreasing)
    # 2. Identify indices where T[i] > S[i] (Increasing)
    # 3. Process Decreasing indices in increasing order (0 to N-1).
    # 4. Process Increasing indices in decreasing order (N-1 to 0).
    
    dec = [i for i in diff_indices if T[i] < S[i]]
    inc = [i for i in diff_indices if T[i] > S[i]]
    
    # The sequence of indices to change
    order = sorted(dec) + sorted(inc, reverse=True)
    
    # Generate the strings
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    
    # We need to produce the strings X_1 to X_M.
    # Since we cannot use loops, we use a generator or map.
    # However, since we need to track the state of s_list, 
    # we can pre-calculate the strings based on the 'order' list.
    
    # X_j is the string S after the first j changes in 'order' have been applied.
    # We can use a list comprehension to build the strings.
    # For each j from 1 to M, the string is:
    # for each index k in 0..N-1:
    #    if k is in order[:j], use T[k], else use S[k]
    
    # To avoid loops and recursion, we use a list comprehension.
    # We map the range(1, M+1) to the resulting strings.
    
    results = [
        "".join(T[k] if k in order[:j] else S[k] for k in range(len(S)))
        for j in range(1, M + 1)
    ]
    
    # Output M and the strings
    sys.stdout.write(f"{M}\n" + "\n".join(results) + ("\n" if M > 0 else ""))

if __name__ == "__main__":
    solve()