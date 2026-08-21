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
    
    # To get the lexicographically smallest array X, we need to change 
    # characters at indices that make the resulting string smaller first.
    # A change at index i makes the string smaller if T[i] < S[i].
    # We should process all indices where T[i] < S[i] in increasing order of i,
    # then all indices where T[i] > S[i] in decreasing order of i.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing group ascending (to affect leftmost chars first)
    # Sort increasing group descending (to affect rightmost chars first, 
    # keeping the prefix as small as possible for as long as possible)
    # Wait, the logic for lexicographical smallest array:
    # We want X_1 to be as small as possible. 
    # To make X_1 small, we should change the first index i where S[i] != T[i] 
    # IF T[i] < S[i]. If T[i] > S[i], changing it makes the string larger.
    # So: 1. Change all i where T[i] < S[i] in increasing order of i.
    # 2. Change all i where T[i] > S[i] in decreasing order of i.
    
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Use reduce to generate the sequence of strings
    # accumulator is (current_string, list_of_strings)
    def apply_change(acc, idx):
        curr_s, history = acc
        # Create new string by replacing character at idx
        new_s = curr_s[:idx] + T[idx] + curr_s[idx+1:]
        return (new_s, history + [new_s])
    
    result = reduce(apply_change, order, (S, []))
    final_history = result[1]
    
    # Output the number of elements and the elements themselves
    print(len(final_history))
    if final_history:
        print('\n'.join(final_history))

if __name__ == "__main__":
    solve()