import sys
from itertools import product

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_orig = input_data[0]
    t_orig = input_data[1]
    
    n = len(s_orig)
    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if s_orig[i] != t_orig[i]]
    m = len(diff_indices)
    
    # To get the lexicographically smallest array X:
    # We must change characters one by one.
    # For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    # makes the string lexicographically smaller immediately.
    # If T[i] > S[i], changing S[i] to T[i] makes it lexicographically larger.
    # Therefore, we should process all indices where T[i] < S[i] first,
    # and among those, process them from left to right (index 0 to n-1).
    # Then, process all indices where T[i] > S[i], but from right to left
    # (index n-1 down to 0), because changing a later character to a larger
    # value is "better" (smaller) than changing an earlier character to a larger value.
    
    # Indices where T[i] < S[i], sorted ascending
    decreasing = sorted([i for i in diff_indices if t_orig[i] < s_orig[i]])
    # Indices where T[i] > S[i], sorted descending
    increasing = sorted([i for i in diff_indices if t_orig[i] > s_orig[i]], reverse=True)
    
    # The sequence of indices to change
    change_order = decreasing + increasing
    
    # Generate the sequence of strings
    current_s = list(s_orig)
    x = []
    for idx in change_order:
        current_s[idx] = t_orig[idx]
        x.append("".join(current_s))
    
    # Output the number of elements and the strings
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()