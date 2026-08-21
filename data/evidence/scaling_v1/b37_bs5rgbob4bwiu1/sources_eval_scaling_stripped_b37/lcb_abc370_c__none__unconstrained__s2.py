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
    #    Therefore, we should process indices where T[i] < S[i] first,
    #    and process them from left to right (smallest index first).
    # 3. For indices where T[i] > S[i], changing S[i] to T[i] 
    #    makes the string lexicographically larger. To keep the sequence
    #    lexicographically smallest, we should process these last,
    #    and process them from right to left (largest index first).
    
    # Indices where T[i] < S[i], sorted ascending
    first_batch = sorted([i for i in diff_indices if t[i] < s[i]])
    # Indices where T[i] > S[i], sorted descending
    second_batch = sorted([i for i in diff_indices if t[i] > s[i]], reverse=True)
    
    # The sequence of indices to change
    change_order = first_batch + second_batch
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating the 
    # state of the list 's' for each index in change_order.
    # Since we need the state after each change, we can track the changes.
    
    # To avoid mutating s in a way that comprehension can't track, 
    # we use a helper function or a loop.
    
    results = []
    current_s = list(s_str)
    for idx in change_order:
        current_s[idx] = t[idx]
        results.append("".join(current_s))
        
    # Output the number of operations
    print(len(results))
    # Output each string in the sequence
    for res in results:
        print(res)

if __name__ == "__main__":
    solve()