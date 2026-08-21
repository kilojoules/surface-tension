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
    # To make the array X lexicographically smallest:
    # 1. We should prioritize changing indices that make the resulting string 
    #    lexicographically smaller first.
    # 2. A change at index i makes the string smaller if T[i] < S[i].
    # 3. A change at index i makes the string larger if T[i] > S[i].
    # Therefore, we process all indices where T[i] < S[i] in increasing order of i,
    # then all indices where T[i] > S[i] in decreasing order of i.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing group ascending by index, and increasing group descending by index
    # Wait, the logic for lexicographical smallest array:
    # If T[i] < S[i], changing S[i] to T[i] makes the string smaller. We want this ASAP.
    # Among those, changing the leftmost index first is best.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger. We want this as late as possible.
    # Among those, changing the rightmost index first is best to keep the prefix smaller for longer.
    
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
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