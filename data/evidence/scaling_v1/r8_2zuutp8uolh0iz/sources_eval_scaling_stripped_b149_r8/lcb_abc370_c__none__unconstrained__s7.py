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
    
    # Indices where the target character is smaller than the current (Priority 1)
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    # Indices where the target character is larger than the current (Priority 2)
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (left to right)
    # Sort increasing indices descending (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    X = []
    for idx in order:
        s_list[idx] = T[idx]
        X.append("".join(s_list))
    
    # Output the number of operations and the strings in X
    print(len(X))
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()