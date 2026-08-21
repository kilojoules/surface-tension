import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To get the lexicographically smallest array X:
    # 1. We want the strings in X to be as small as possible as early as possible.
    # 2. If S[i] > T[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # 3. If S[i] < T[i], changing S[i] to T[i] makes the string lexicographically larger.
    # Therefore, we should process all indices where S[i] > T[i] first (in increasing order of i),
    # and then all indices where S[i] < T[i] (in decreasing order of i).
    # Wait, the rule for the array X is: X_j is compared to Y_j. 
    # To make X_1 smallest, we want to change the first possible index that decreases the string.
    # Actually, the simplest greedy strategy:
    # First, handle all i where S[i] > T[i] in increasing order of i.
    # Then, handle all i where S[i] < T[i] in decreasing order of i.
    # This ensures we reduce the string as much as possible as early as possible.
    
    decreasing = [i for i in diff_indices if S[i] > T[i]]
    increasing = [i for i in diff_indices if S[i] < T[i]]
    
    # Sort decreasing indices ascending (change leftmost character that decreases first)
    # Sort increasing indices descending (change rightmost character that increases first)
    # to keep the prefix as small as possible for as long as possible.
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Use reduce to simulate the transformation process and collect the sequence of strings
    # acc is the (current_string, list_of_strings)
    def transform(acc, idx):
        curr_s, history = acc
        # Create new string by replacing character at idx
        new_s = curr_s[:idx] + T[idx] + curr_s[idx+1:]
        return (new_s, history + [new_s])
    
    result = reduce(transform, order, (S, []))
    X = result[1]
    
    # Output the number of elements and the elements themselves
    print(len(X))
    sys.stdout.write('\n'.join(X) + ('\n' if X else ''))

if __name__ == "__main__":
    solve()