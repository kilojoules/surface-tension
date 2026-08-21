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
    
    # To get the lexicographically smallest array X, we want the strings 
    # in X to be as small as possible at each step.
    # A string is smaller if its characters at earlier positions are smaller.
    # Therefore, we should prioritize changing characters at the earliest 
    # possible indices that make the string smaller, or delaying changes 
    # that make the string larger.
    
    # Specifically, for each index i where S[i] != T[i]:
    # If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    # To keep the sequence X lexicographically smallest, we should perform all 
    # "reducing" changes first (from left to right) and then all "increasing" 
    # changes (from right to left).
    # Wait, the rule is simpler: to make the resulting strings in X smallest,
    # we should process indices i where T[i] < S[i] from left to right,
    # and indices i where T[i] > S[i] from right to left.
    
    reducing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort reducing indices ascending to make the string smaller as early as possible.
    # Sort increasing indices descending to delay the "increase" at early positions.
    order = sorted(reducing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    current_S = list(S)
    X = []
    for idx in order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output the number of elements and the elements themselves
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()