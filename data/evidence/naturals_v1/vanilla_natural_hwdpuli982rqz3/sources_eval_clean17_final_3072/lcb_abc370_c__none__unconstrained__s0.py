import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To get the lexicographically smallest array X, we want each X_j to be 
    # as small as possible. 
    # We should process the characters of S from left to right.
    # For each index i where S[i] != T[i]:
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    # To minimize the sequence lexicographically, we should first handle all 
    # indices where T[i] < S[i] (from left to right) to decrease the string 
    # as quickly as possible, then handle indices where T[i] > S[i] (from right to left)
    # so that the "larger" changes happen as late as possible in the string.
    
    # Correction on strategy:
    # To make X_1 smallest: we want the smallest possible character at the earliest possible position.
    # If there is any i such that T[i] < S[i], picking the smallest such i and changing S[i] to T[i]
    # is optimal.
    # If for all i, T[i] >= S[i], we want to delay the increase. To keep X_1 smallest,
    # we should change the rightmost index i where T[i] > S[i].
    
    # Let's refine:
    # 1. Find all i where S[i] > T[i]. Sort them ascending.
    # 2. Find all i where S[i] < T[i]. Sort them descending.
    # The sequence of indices to change is (1) followed by (2).
    
    decrease_indices = [i for i in range(N) if S[i] > T[i]]
    increase_indices = [i for i in range(N) if S[i] < T[i]]
    
    # Sort decrease_indices ascending to make the string smaller as early as possible.
    # Sort increase_indices descending to make the string larger as late as possible.
    order = decrease_indices + sorted(increase_indices, reverse=True)
    
    # Convert S to a list to allow mutations
    s_list = list(S)
    
    # Generate the sequence X
    X = []
    for idx in order:
        s_list[idx] = T[idx]
        X.append("".join(s_list))
    
    # Output results
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()