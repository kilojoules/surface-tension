import sys

def solve():
    # Read S and T from standard input
    try:
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    if not s or not t:
        # Handle cases where input might be empty or missing
        if s == t:
            print(0)
            return
        # This case shouldn't happen based on constraints
        return

    n = len(s)
    diff_indices = []
    for i in range(n):
        if s[i] != t[i]:
            diff_indices.append(i)

    # The minimum number of elements M is the number of positions where S and T differ.
    # To make the array X lexicographically smallest, we must prioritize changes 
    # that make the string X_i lexicographically smaller.
    # A change at index i makes the string smaller if S[i] > T[i] (we decrease the char).
    # A change at index i makes the string larger if S[i] < T[i] (we increase the char).
    
    # Therefore, we should process all indices where S[i] > T[i] first, 
    # and among those, process them in increasing order of index (from left to right).
    # Then, we process all indices where S[i] < T[i], but in decreasing order of index
    # (from right to left), because changing a character to something larger at 
    # the leftmost position makes the string larger immediately.
    
    decreasing = [] # Indices where S[i] > T[i]
    increasing = [] # Indices where S[i] < T[i]
    
    for i in diff_indices:
        if s[i] > t[i]:
            decreasing.append(i)
        else:
            increasing.append(i)
            
    # Sort decreasing indices in ascending order (left to right)
    decreasing.sort()
    # Sort increasing indices in descending order (right to left)
    increasing.sort(reverse=True)
    
    order = decreasing + increasing
    
    print(len(order))
    
    current_s = list(s)
    for idx in order:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()