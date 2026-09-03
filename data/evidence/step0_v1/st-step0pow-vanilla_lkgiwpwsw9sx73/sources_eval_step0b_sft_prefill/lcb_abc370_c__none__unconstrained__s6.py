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
    
    # To get the lexicographically smallest array X, we should change characters
    # that result in the lexicographically smallest string at each step.
    # A string A is smaller than B if at the first differing position, A has a smaller char.
    # To make S smaller quickly, we should prioritize changing characters at smaller indices
    # if the target character T[i] is smaller than the current S[i].
    # If T[i] is larger than S[i], we should delay that change as long as possible
    # to keep the string smaller for longer.
    
    # Specifically:
    # 1. Indices i where T[i] < S[i] should be processed in increasing order (left to right).
    # 2. Indices i where T[i] > S[i] should be processed in decreasing order (right to left).
    
    # This ensures that we "drop" the characters to smaller ones as early as possible
    # and "raise" the characters to larger ones as late as possible.
    
    decreasing_indices = [i for i in diff_indices if T[i] < S[i]]
    increasing_indices = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending, increasing indices descending
    decreasing_indices.sort()
    increasing_indices.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing_indices + increasing_indices
    
    # Generate the sequence of strings
    # We use a list for S since strings are immutable
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