import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To get the lexicographically smallest array X, we want each X_j to be 
    # as small as possible. 
    # X_j is formed by changing one character of X_{j-1}.
    # To make X_1 smallest, we should look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], we want to delay this change as much as possible to keep 
    # the prefix small, but we must change it eventually.
    
    # Correct Strategy for lexicographical minimality of the sequence:
    # 1. First, process all indices i where T[i] < S[i] in increasing order of i.
    #    This reduces the string at the earliest possible position.
    # 2. Then, process all indices i where T[i] > S[i] in decreasing order of i.
    #    This delays the increase of the string to the latest possible position.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (to hit the leftmost character first)
    # Sort increasing indices descending (to hit the rightmost character first)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    # We use a list of characters because strings are immutable in Python
    current_S = list(S)
    X = []
    
    for idx in order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output results
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()