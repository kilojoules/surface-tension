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
    # 2. We should prioritize changing indices that make the resulting string 
    #    lexicographically smaller first.
    #    - If T[i] < S[i], changing S[i] to T[i] makes the string smaller.
    #      These should be done as early as possible, from left to right.
    #    - If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    #      These should be done as late as possible, from right to left.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (left to right)
    # Sort increasing indices descending (right to left)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list of characters for S since strings are immutable
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