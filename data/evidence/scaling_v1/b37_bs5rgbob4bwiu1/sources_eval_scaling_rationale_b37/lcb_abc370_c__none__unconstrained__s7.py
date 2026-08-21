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
    
    # To get the lexicographically smallest sequence of strings:
    # 1. If S[i] > T[i], we want to change it as early as possible 
    #    because decreasing a character makes the string lexicographically smaller.
    # 2. If S[i] < T[i], we want to change it as late as possible 
    #    because increasing a character makes the string lexicographically larger.
    # 3. For those in the same category, we process them from left to right (index order).
    
    # Category 1: S[i] > T[i] (Decrease) - Process in increasing order of index
    decrease = [i for i in diff_indices if S[i] > T[i]]
    # Category 2: S[i] < T[i] (Increase) - Process in decreasing order of index
    increase = sorted([i for i in diff_indices if S[i] < T[i]], reverse=True)
    
    # The sequence of indices to change
    order = decrease + increase
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a list of strings (X).
    # The current value is the index to change.
    def transform(acc, idx):
        current_s = acc[-1] if acc else S
        # Strings are immutable, so we create a new one
        new_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [new_s]

    # Initialize reduce with an empty list. 
    # Since we need the state of S, we handle the first step carefully.
    # We use a helper to manage the string state across the reduction.
    
    # Correct approach with reduce: 
    # Accumulator stores (current_string, list_of_strings)
    result = reduce(
        lambda state, idx: (
            state[0][:idx] + T[idx] + state[0][idx+1:], 
            state[1] + [state[0][:idx] + T[idx] + state[0][idx+1:]]
        ),
        order,
        (S, [])
    )
    
    X = result[1]
    
    # Output the number of elements and each string
    print(len(X))
    if X:
        print('\n'.join(X))

if __name__ == "__main__":
    solve()