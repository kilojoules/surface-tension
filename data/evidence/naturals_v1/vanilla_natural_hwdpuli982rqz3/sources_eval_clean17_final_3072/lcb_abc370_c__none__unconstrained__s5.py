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
    
    # To get the lexicographically smallest array X, we want the strings X_i 
    # to be as small as possible. 
    # We should process the indices in an order that minimizes the resulting strings.
    # Specifically, if we change S[i] to T[i], and T[i] < S[i], we want to do this 
    # as early as possible to make the string smaller.
    # If T[i] > S[i], we want to do this as late as possible to avoid making 
    # the string larger prematurely.
    
    # Indices where T[i] < S[i] (decreasing) sorted ascending
    # Indices where T[i] > S[i] (increasing) sorted descending
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending to make the string smaller at the earliest possible position
    decreasing.sort()
    # Sort increasing indices descending to delay making the string larger at the earliest possible position
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list of characters because strings are immutable in Python
    current_S = list(S)
    X = []
    for i in order:
        current_S[i] = T[i]
        X.append("".join(current_S))
    
    # Output results
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()