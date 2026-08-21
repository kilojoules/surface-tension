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
    
    # To get the lexicographically smallest array X:
    # We must change characters one by one.
    # For a specific index i, if T[i] < S[i], changing S[i] to T[i] 
    # makes the string lexicographically smaller immediately.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    # Therefore, we should process all indices where T[i] < S[i] first (in increasing order of i),
    # and then process all indices where T[i] > S[i] (in decreasing order of i).
    
    # Indices where the target character is smaller than the current character
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    # Indices where the target character is larger than the current character
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending to make the string as small as possible as early as possible
    # Sort increasing indices descending so that the "increase" happens as late as possible 
    # at the rightmost positions first.
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    s_list = list(S)
    X = [ "".join(s_list) for i in order if (s_list.__setitem__(i, T[i]) or True) ]
    
    # The logic inside the list comprehension is a trick to perform the assignment 
    # and capture the state. However, since __setitem__ returns None, 
    # we use a helper function or a loop for clarity and correctness.
    
def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    S, T = input_data[0], input_data[1]
    
    diffs = [i for i in range(len(S)) if S[i] != T[i]]
    # Priority 1: Indices where T[i] < S[i], processed left-to-right
    # Priority 2: Indices where T[i] > S[i], processed right-to-left
    order = sorted([i for i in diffs if T[i] < S[i]]) + \
            sorted([i for i in diffs if T[i] > S[i]], reverse=True)
    
    res = []
    current_s = list(S)
    for i in order:
        current_s[i] = T[i]
        res.append("".join(current_s))
    
    print(len(res))
    for s in res:
        print(s)

if __name__ == "__main__":
    solve_final()