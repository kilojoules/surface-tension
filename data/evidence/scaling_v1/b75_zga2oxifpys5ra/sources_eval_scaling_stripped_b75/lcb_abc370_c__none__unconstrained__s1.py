import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]
    
    # Identify indices where S and T differ
    # To make the array lexicographically smallest, we must prioritize 
    # changing characters at indices that make the resulting string smaller.
    # If S[i] > T[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If S[i] < T[i], changing S[i] to T[i] makes the string lexicographically larger.
    # Therefore, we should process all indices i where S[i] > T[i] first (from left to right),
    # and then all indices i where S[i] < T[i] (from right to left).
    
    diff_indices = [i for i in range(len(s)) if s[i] != t[i]]
    
    # Indices where S[i] > T[i] (decreasing the string value)
    # Process these from left to right to minimize the string as early as possible.
    decreasing = [i for i in diff_indices if s[i] > t[i]]
    
    # Indices where S[i] < T[i] (increasing the string value)
    # Process these from right to left so the prefix remains smaller for longer.
    increasing = [i for i in diff_indices if s[i] < t[i]]
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing + increasing
    
    # Generate the sequence of strings
    # We use a list and join to simulate the mutation of S
    s_list = list(s)
    x = [
        "".join(s_list[:i] + [t[i]] + s_list[i+1:]) 
        for i in order
        for s_list in [ (s_list[:i] + [t[i]] + s_list[i+1:]) ]
    ]
    
    # Since the list comprehension above is tricky with state, 
    # let's use a generator/loop to build X.
    
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    s, t = input_data[0], input_data[1]
    
    diffs = [i for i in range(len(s)) if s[i] != t[i]]
    # 1. Indices where S[i] > T[i], sorted ascending
    # 2. Indices where S[i] < T[i], sorted descending
    order = sorted([i for i in diffs if s[i] > t[i]]) + \
            sorted([i for i in diffs if s[i] < t[i]], reverse=True)
    
    # Construct the strings
    res = []
    current_s = list(s)
    for i in order:
        current_s[i] = t[i]
        res.append("".join(current_s))
        
    print(len(res))
    for string in res:
        print(string)

if __name__ == "__main__":
    solve_final()