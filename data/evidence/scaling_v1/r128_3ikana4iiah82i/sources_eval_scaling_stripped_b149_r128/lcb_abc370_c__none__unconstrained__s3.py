import sys
from itertools import product

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    M = len(diff_indices)

    # We need to change characters at diff_indices one by one.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in X to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to the smallest possible values.
    # However, we must eventually reach T. 
    # The most restrictive constraint is that we change one character per step.
    # To keep X_1 smallest, we should change the first differing index 
    # to the smallest character that helps us reach T, which is T[diff_indices[0]].
    # But wait, if T[diff_indices[0]] is larger than S[diff_indices[0]], 
    # we might want to change a later index first if that makes the string smaller.
    # Actually, the rule is: we must change one character per step.
    # To make X_1 smallest, we look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], changing any S[j] to T[j] for j > i will keep the 
    # prefix S[0...i] the same, but the string will be larger than if we 
    # could have decreased S[i].
    # Since we can only change characters to their target values in T to 
    # minimize the number of steps (M), the only choice is the order of indices.
    
    # To minimize X lexicographically:
    # We should process indices in an order that keeps the string as small as possible.
    # 1. If T[i] < S[i], we want to perform this change as early as possible.
    # 2. If T[i] > S[i], we want to perform this change as late as possible.
    # Among those that must be changed early, the one with the smallest index is most impactful.
    # Among those that must be changed late, the one with the largest index is least impactful.
    
    # Correct Strategy:
    # Sort indices i where T[i] < S[i] in increasing order.
    # Sort indices i where T[i] > S[i] in decreasing order.
    # The sequence of changes is: all (T[i] < S[i]) indices, then all (T[i] > S[i]) indices.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing (T[i] < S[i]) ascending by index to affect the earliest character first
    decreasing.sort()
    # Sort increasing (T[i] > S[i]) descending by index to affect the latest character first
    increasing.sort(reverse=True)
    
    order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list comprehension to build the strings based on the order of indices
    # For the k-th string in X, indices order[0...k-1] have been updated to T.
    X = [
        "".join([T[i] if i in order[:k] else S[i] for i in range(len(S))])
        for k in range(1, M + 1)
    ]
    
    # Output the result
    print(M)
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()