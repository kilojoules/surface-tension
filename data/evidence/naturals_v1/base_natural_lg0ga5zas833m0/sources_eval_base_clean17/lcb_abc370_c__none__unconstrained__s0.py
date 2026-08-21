import sys

def solve():
    # Read S and T from standard input
    try:
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    if not s or not t:
        # Handle empty input cases if necessary
        if not s and not t:
            print(0)
            return
        # In standard competitive programming, this case is usually not hit
        # given the constraints.

    n = len(s)
    # Identify the indices where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    m = len(diff_indices)
    
    if m == 0:
        print(0)
        return

    # To get the lexicographically smallest array X, we must decide the order
    # in which we change characters from S to T.
    # For a specific index i, if T[i] < S[i], changing S[i] to T[i] makes 
    # the string lexicographically smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    
    # Strategy:
    # 1. First, process all indices where T[i] < S[i] in increasing order of index.
    #    This makes the string smaller as early as possible.
    # 2. Then, process all indices where T[i] > S[i] in decreasing order of index.
    #    This delays making the string larger for as long as possible.
    
    decreasing = [] # indices where T[i] < S[i]
    increasing = [] # indices where T[i] > S[i]
    
    for i in diff_indices:
        if t[i] < s[i]:
            decreasing.append(i)
        else:
            increasing.append(i)
            
    # Sort indices for the "decreasing" phase (T[i] < S[i]) in ascending order
    decreasing.sort()
    # Sort indices for the "increasing" phase (T[i] > S[i]) in descending order
    increasing.sort(reverse=True)
    
    # The sequence of indices to change
    order = decreasing + increasing
    
    print(m)
    current_s = list(s)
    for idx in order:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()