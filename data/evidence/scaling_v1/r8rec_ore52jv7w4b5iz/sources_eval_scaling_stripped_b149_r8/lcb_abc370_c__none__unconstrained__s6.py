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
    
    # To minimize the array lexicographically, we must change characters 
    # in an order that makes the resulting string as small as possible at each step.
    # 1. If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #    We should do these as early as possible, from left to right.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should do these as late as possible, from right to left.
    
    # Group 1: Indices where T[i] < S[i] (processed left to right)
    # Group 2: Indices where T[i] > S[i] (processed right to left)
    
    left_to_right = [i for i in diff_indices if T[i] < S[i]]
    right_to_left = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort indices for Group 1 ascending, and Group 2 descending
    order = sorted(left_to_right) + sorted(right_to_left, reverse=True)
    
    # Generate the sequence of strings
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    X = []
    for idx in order:
        s_list[idx] = T[idx]
        X.append("".join(s_list))
    
    # Output the number of elements and the elements themselves
    print(len(X))
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()