import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    n = len(S)
    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if S[i] != T[i]]
    m = len(diff_indices)
    
    # To minimize the number of elements in X, we must change exactly one 
    # character per step. The minimum number of steps is the number of 
    # positions where S and T differ.
    # To make the array X lexicographically smallest, we want the strings 
    # at the beginning of the array to be as small as possible.
    # This means we should prioritize changing characters at indices that 
    # make the resulting string lexicographically smaller first.
    # Specifically, if T[i] < S[i], changing S[i] to T[i] makes the string smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    
    # Strategy:
    # 1. Process all indices i where T[i] < S[i] in increasing order of i.
    #    (Changing a character to something smaller as early as possible 
    #    in the string makes the string lexicographically smallest).
    # 2. Process all indices i where T[i] > S[i] in decreasing order of i.
    #    (Changing a character to something larger as late as possible 
    #    in the string keeps the string lexicographically smaller for longer).
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing group (T[i] < S[i]) in ascending order of index
    decreasing.sort()
    # Sort increasing group (T[i] > S[i]) in descending order of index
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing + increasing
    
    # Construct the array X
    X = []
    current_S = list(S)
    for idx in order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output the result
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()