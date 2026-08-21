import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # To minimize the number of elements M, we must change only the characters 
    # where S[i] != T[i]. M will be the number of indices i where S[i] != T[i].
    # To make the array X lexicographically smallest, we want X_1 to be as small as possible,
    # then X_2, and so on.
    
    # A character change at index i makes the string smaller if S[i] > T[i] 
    # and larger if S[i] < T[i].
    # To get the lexicographically smallest sequence:
    # 1. We should first process all indices i where S[i] > T[i] from left to right.
    #    Changing S[i] to T[i] when S[i] > T[i] decreases the string lexicographically.
    # 2. Then we process all indices i where S[i] < T[i] from right to left.
    #    Changing S[i] to T[i] when S[i] < T[i] increases the string lexicographically,
    #    so we want to delay these changes as much as possible or do them at the end.
    #    Specifically, changing the rightmost characters first keeps the prefix smaller for longer.
    
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    
    # Indices that decrease the string (S[i] > T[i])
    decreasing = [i for i in diff_indices if S[i] > T[i]]
    # Indices that increase the string (S[i] < T[i])
    increasing = [i for i in diff_indices if S[i] < T[i]]
    
    # Sort decreasing indices ascending (left to right) to drop the string value quickly
    # Sort increasing indices descending (right to left) to raise the string value slowly
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Convert S to a list to allow mutations
    s_list = list(S)
    
    # Generate the sequence X
    X = []
    for i in order:
        s_list[i] = T[i]
        X.append("".join(s_list))
    
    # Output results
    print(len(X))
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()