import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]
    
    s = list(s_str)
    t = list(t_str)
    n = len(s)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    
    # To get the lexicographically smallest sequence of strings:
    # 1. We must change characters one by one.
    # 2. For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    #    makes the string lexicographically smaller immediately.
    #    We should do these changes as early as possible (from left to right).
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should delay these changes as much as possible (from right to left).
    
    # Indices where T[i] < S[i] (processed left to right)
    decreasing = [i for i in diff_indices if t[i] < s[i]]
    # Indices where T[i] > S[i] (processed right to left)
    increasing = [i for i in diff_indices if t[i] > s[i]]
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    change_order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the 
    # current state of S for each index in the change_order.
    # Since we need the state after each change, we can track the changes.
    
    # To avoid loops, we can use a helper function or a list comprehension 
    # that builds the strings based on the index of the change.
    
    # For the i-th change (0-indexed), the character at change_order[i] becomes T[change_order[i]],
    # and all characters at change_order[j] for j < i have already been changed.
    
    res = [
        "".join([t[k] if k in change_order[:i+1] else s[k] for k in range(n)])
        for i in range(len(change_order))
    ]
    
    # Output the number of operations
    print(len(res))
    # Output each string in the sequence
    for string in res:
        print(string)

if __name__ == "__main__":
    solve()