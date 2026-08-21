import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    n = len(S)
    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if S[i] != T[i]]
    m = len(diff_indices)
    
    # To minimize the number of elements in X, we must change exactly one 
    # character per step. The minimum number of steps is the number of 
    # positions where S and T differ.
    # To make the array X lexicographically smallest, at each step we want 
    # the resulting string to be as small as possible.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to their target values in T, BUT only if the target 
    # character T[i] is smaller than the current character S[i].
    # If T[i] is larger than S[i], we want to delay that change as long 
    # as possible to keep the string lexicographically smaller for longer.
    
    # Divide indices into two groups:
    # 1. Indices where T[i] < S[i] (improving the string immediately)
    # 2. Indices where T[i] > S[i] (worsening the string)
    
    decreasing = [] # T[i] < S[i]
    increasing = [] # T[i] > S[i]
    
    for i in diff_indices:
        if T[i] < S[i]:
            decreasing.append(i)
        else:
            increasing.append(i)
            
    # For decreasing indices, we want to process them from left to right
    # to make the string smaller as early as possible.
    decreasing.sort()
    
    # For increasing indices, we want to process them from right to left
    # to keep the prefix of the string smaller for as long as possible.
    increasing.sort(reverse=True)
    
    # The sequence of indices to change is all 'decreasing' then all 'increasing'
    change_order = decreasing + increasing
    
    # Construct the array X
    X = []
    current_S = list(S)
    for idx in change_order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
        
    # Output the result
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()