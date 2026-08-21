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
    
    # To get the lexicographically smallest array X:
    # 1. We must change characters one by one.
    # 2. For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    #    makes the string lexicographically smaller immediately.
    #    These should be handled first, from left to right.
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string 
    #    lexicographically larger. These should be handled last, 
    #    from right to left, to keep the prefix as small as possible for as long as possible.
    
    # Group 1: T[i] < S[i] (processed left to right)
    # Group 2: T[i] > S[i] (processed right to left)
    
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
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()