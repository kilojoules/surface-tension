import sys

def solve():
    # Read input strings S and T
    try:
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    if not s or not t:
        # Handle potential empty input cases
        if not s and not t:
            print(0)
            return
        return

    n = len(s)
    # Find indices where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    m = len(diff_indices)
    
    # If S already equals T, M = 0
    if m == 0:
        print(0)
        return

    # To get the lexicographically smallest sequence of strings X:
    # 1. We must change characters one by one (minimum M elements).
    # 2. To make the strings X_j lexicographically smallest, we should prioritize 
    #    changing characters at indices that make the string smaller.
    # 3. A character change at index i makes the string smaller if T[i] < S[i].
    #    If T[i] > S[i], changing it makes the string larger.
    
    # Strategy:
    # First, process all indices i where T[i] < S[i] in increasing order of i.
    # This reduces the string as early as possible.
    # Then, process all indices i where T[i] > S[i] in decreasing order of i.
    # This delays the increase of the string as much as possible.
    
    decreasing = [] # indices where T[i] < S[i]
    increasing = [] # indices where T[i] > S[i]
    
    for i in diff_indices:
        if t[i] < s[i]:
            decreasing.append(i)
        else:
            increasing.append(i)
            
    # Sort decreasing indices in increasing order (left to right)
    decreasing.sort()
    # Sort increasing indices in decreasing order (right to left)
    increasing.sort(reverse=True)
    
    order = decreasing + increasing
    
    # Construct the array X
    current_s = list(s)
    result = []
    for idx in order:
        current_s[idx] = t[idx]
        result.append("".join(current_s))
        
    # Output the number of elements M
    print(len(result))
    # Output each string in the sequence
    for string in result:
        print(string)

if __name__ == "__main__":
    solve()