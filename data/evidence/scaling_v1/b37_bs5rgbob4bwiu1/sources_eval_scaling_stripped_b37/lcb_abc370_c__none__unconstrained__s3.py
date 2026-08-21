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
    # 1. We must change characters at indices where S[i] > T[i] first.
    #    For these, we want to change them as early as possible to decrease 
    #    the string value, but the problem asks for the lexicographically 
    #    smallest ARRAY. 
    #    Wait, if S[i] > T[i], changing it to T[i] immediately makes the 
    #    string smaller. To keep the array lexicographically smallest, 
    #    we should process indices i where S[i] > T[i] in increasing order of i.
    # 2. For indices where S[i] < T[i], changing them makes the string larger.
    #    To keep the array lexicographically smallest, we should process 
    #    these in decreasing order of i.
    
    # Indices where S[i] > T[i], sorted ascending
    decreasing = sorted([i for i in diff_indices if s[i] > t[i]])
    # Indices where S[i] < T[i], sorted descending
    increasing = sorted([i for i in diff_indices if s[i] < t[i]], reverse=True)
    
    # The sequence of indices to change
    change_order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list comprehension to create the state of S after each change
    # Since we cannot mutate S inside a comprehension, we use a helper function 
    # or a loop to generate the sequence.
    
    res = []
    current_s = list(s_str)
    for idx in change_order:
        current_s[idx] = t[idx]
        res.append("".join(current_s))
        
    # Output the number of operations
    print(len(res))
    # Output each string in the sequence
    for string in res:
        print(string)

if __name__ == "__main__":
    solve()