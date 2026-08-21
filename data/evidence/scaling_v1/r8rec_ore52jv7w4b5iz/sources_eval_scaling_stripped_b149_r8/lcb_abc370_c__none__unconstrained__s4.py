import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To minimize the number of elements in X, we must change exactly one 
    # character per step for each index where S[i] != T[i].
    # To make the array X lexicographically smallest, we must prioritize 
    # changing characters that make the resulting string smaller.
    
    # A change at index i makes the string smaller if T[i] < S[i].
    # A change at index i makes the string larger if T[i] > S[i].
    # We should process all indices where T[i] < S[i] first, in increasing order of i.
    # Then process all indices where T[i] > S[i], in decreasing order of i.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending to affect the leftmost character first
    decreasing.sort()
    # Sort increasing indices descending to affect the rightmost character first
    # (because changing a left character to something larger makes the string 
    # lexicographically larger immediately; we delay that as much as possible).
    increasing.sort(reverse=True)
    
    order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    X = []
    for i in order:
        s_list[i] = T[i]
        X.append("".join(s_list))
    
    # Output the number of elements and the elements themselves
    print(len(X))
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()