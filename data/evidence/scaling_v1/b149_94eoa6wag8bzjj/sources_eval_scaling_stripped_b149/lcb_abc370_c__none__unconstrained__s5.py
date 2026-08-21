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
    # character per step. The minimum number of steps is the number of 
    # indices where S[i] != T[i].
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in X to be lexicographically smaller.
    # This means we should prioritize changing characters at indices that 
    # make the string smaller as early as possible.
    # Specifically, if T[i] < S[i], changing S[i] to T[i] makes the string 
    # lexicographically smaller. We should do these changes from left to right.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger. 
    # We should do these changes from right to left to keep the prefix 
    # smaller for as long as possible.
    
    # Indices that decrease the string value (T[i] < S[i])
    decrease = [i for i in diff_indices if T[i] < S[i]]
    # Indices that increase the string value (T[i] > S[i])
    increase = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decrease indices ascending (left to right)
    # Sort increase indices descending (right to left)
    order = sorted(decrease) + sorted(increase, reverse=True)
    
    # Generate the sequence of strings
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