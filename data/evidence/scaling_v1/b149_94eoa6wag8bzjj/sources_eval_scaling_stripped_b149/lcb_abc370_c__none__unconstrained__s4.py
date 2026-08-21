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
    # If T[i] > S[i], changing it makes the string lexicographically larger,
    # so we should delay those changes as much as possible.
    
    # Correct strategy for lexicographical smallest array:
    # 1. Change indices i where T[i] < S[i] first, in increasing order of i.
    # 2. Change indices i where T[i] > S[i] last, in decreasing order of i.
    # Wait, the rule is: we want X_1 to be as small as possible.
    # To make X_1 small, we should change the first index i where S[i] != T[i]
    # IF T[i] < S[i]. If T[i] > S[i], changing it makes the string larger.
    # Actually, the simplest greedy approach to make the sequence of strings 
    # lexicographically smallest is:
    # Always pick the index i from the remaining diff_indices that results in 
    # the lexicographically smallest string.
    # The best index i is the smallest i such that T[i] < S[i].
    # If no such i exists, pick the largest i such that T[i] > S[i].
    
    # Let's refine:
    # We have a set of indices that must be changed.
    # In each step, we choose one index i and set S[i] = T[i].
    # To make the resulting string smallest:
    # - We prefer changing an index i where T[i] < S[i] over T[i] > S[i].
    # - Among indices where T[i] < S[i], we want to change the smallest index i first.
    # - Among indices where T[i] > S[i], we want to change the largest index i first
    #   (because changing a small index to a larger character increases the 
    #   string's lexicographical value more than changing a large index).
    
    # Let's categorize indices
    smaller = [i for i in diff_indices if T[i] < S[i]]
    larger = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort 'smaller' ascending to fix the leftmost characters first.
    # Sort 'larger' descending to fix the rightmost characters first.
    order = sorted(smaller) + sorted(larger, reverse=True)
    
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