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
    # character per step. The minimum number of steps is the number of differences.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in the array to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to the target characters in T, BUT only if the 
    # target character T[i] is smaller than the current character S[i].
    # Actually, the rule is simpler: to make the resulting string smallest,
    # we should process indices i where T[i] < S[i] first (from left to right),
    # and then process indices i where T[i] > S[i] (from right to left).
    # Wait, that's for a different problem. Let's re-evaluate.
    
    # For the array X to be lexicographically smallest:
    # We want X_1 to be as small as possible, then X_2, etc.
    # To make X_1 small, we should change the first index i where S[i] != T[i]
    # IF T[i] < S[i]. If T[i] > S[i], changing it makes the string larger.
    # However, we MUST change one character per step.
    # The optimal strategy to keep the strings lexicographically smallest:
    # 1. Change indices i where T[i] < S[i] in increasing order of i.
    # 2. Change indices i where T[i] > S[i] in decreasing order of i.
    
    # Let's refine:
    # We want to reach T from S. In each step, we change one S[i] to T[i].
    # To keep the sequence of strings X lexicographically smallest:
    # - If we can make the string smaller, we should do it as early as possible (smallest index).
    # - If we must make the string larger, we should delay it as much as possible (largest index).
    
    decrease_indices = [i for i in diff_indices if T[i] < S[i]]
    increase_indices = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decrease indices ascending, increase indices descending
    order = sorted(decrease_indices) + sorted(increase_indices, reverse=True)
    
    # Generate the sequence of strings
    current_S = list(S)
    X = []
    for i in order:
        current_S[i] = T[i]
        X.append("".join(current_S))
    
    # Output the result
    print(len(X))
    for s in X:
        print(s)

if __name__ == "__main__":
    solve()