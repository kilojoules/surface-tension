import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To get the lexicographically smallest array X, we want the strings 
    # appearing earlier in X to be lexicographically smaller.
    # This means we should prioritize changing characters at the beginning of the string
    # to their target values in T if the target is smaller than the current,
    # or handle them in a specific order to minimize the resulting strings.
    
    # Specifically:
    # 1. If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #    We should do these as early as possible, from left to right.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should do these as late as possible, from right to left.
    
    # Sort indices where T[i] < S[i] in ascending order
    # Sort indices where T[i] > S[i] in descending order
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # The optimal sequence of indices to change is:
    # All indices i where T[i] < S[i] (from left to right),
    # followed by all indices i where T[i] > S[i] (from right to left).
    
    # Note: The problem asks for the lexicographically smallest ARRAY.
    # If we change S[i] to T[i], the new string is X_k.
    # To make X_1 smallest, we want the first character that differs from S to decrease.
    # So we pick the smallest i such that T[i] < S[i].
    # After that, we repeat.
    # Once no more T[i] < S[i] exist, we must pick the largest i such that T[i] > S[i]
    # to keep the prefix of the string as small as possible for as long as possible.
    
    order = sorted([i for i in diff_indices if T[i] < S[i]]) + \
            sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)
    
    # Convert S to a list to allow mutation
    s_list = list(S)
    
    # Generate the sequence of strings
    results = []
    for idx in order:
        s_list[idx] = T[idx]
        results.append("".join(s_list))
    
    # Output results
    print(M)
    for res in results:
        print(res)

if __name__ == "__main__":
    solve()