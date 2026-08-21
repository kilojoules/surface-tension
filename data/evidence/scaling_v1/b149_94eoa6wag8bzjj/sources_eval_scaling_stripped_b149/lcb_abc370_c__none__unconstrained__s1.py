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
    # character per step. The minimum number of steps is the number of differences.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in the array to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to the target characters in T, BUT only if the 
    # target character T[i] is smaller than the current character S[i].
    # Actually, the rule is simpler: to make the string X_j smallest, 
    # we should change the leftmost index i where S[i] != T[i] if T[i] < S[i].
    # If for all differing indices i, T[i] > S[i], we must change the 
    # rightmost index first to keep the prefix as small as possible for as long as possible.
    
    # Correct Strategy for Lexicographical Smallest Array:
    # 1. If there are indices i where T[i] < S[i], we should process all such 
    #    indices from left to right first.
    # 2. Then, we process all indices i where T[i] > S[i] from right to left.
    
    left_to_right = [i for i in diff_indices if T[i] < S[i]]
    right_to_left = [i for i in diff_indices if T[i] > S[i]][::-1]
    
    # The sequence of indices to change
    order = left_to_right + right_to_left
    
    # Generate the array X
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    X = []
    for idx in order:
        s_list[idx] = T[idx]
        X.append("".join(s_list))
    
    # Output the result
    print(len(X))
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()