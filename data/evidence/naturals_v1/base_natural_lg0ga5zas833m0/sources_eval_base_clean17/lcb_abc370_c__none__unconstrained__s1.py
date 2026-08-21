import sys

def solve():
    # Read S and T from standard input
    try:
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    if not s or not t:
        # Handle potential empty inputs if constraints are violated
        if s == t:
            print(0)
            return
        else:
            return

    n = len(s)
    
    # The minimum number of elements M is the number of indices i where S[i] != T[i].
    # To make the array X lexicographically smallest, we need to determine 
    # the order in which we change the characters.
    
    # Let's identify all indices where S and T differ.
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    m = len(diff_indices)
    
    if m == 0:
        print(0)
        return

    # To make the resulting sequence of strings X lexicographically smallest:
    # For each index i where S[i] != T[i]:
    # If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    
    # Strategy:
    # 1. First, process all indices i where T[i] < S[i].
    #    Among these, we should process them from left to right (smallest index first)
    #    because a change at a smaller index has a larger impact on the string's 
    #    lexicographical value.
    # 2. Then, process all indices i where T[i] > S[i].
    #    Among these, we should process them from right to left (largest index first).
    #    Why? Because changing a character to something larger increases the string's
    #    value. To keep the sequence of strings smallest, we want the "increase" to
    #    happen at the least significant position (the rightmost) first.
    
    decreasing = [] # Indices where T[i] < S[i]
    increasing = [] # Indices where T[i] > S[i]
    
    for i in diff_indices:
        if t[i] < s[i]:
            decreasing.append(i)
        else:
            increasing.append(i)
            
    # Sort decreasing indices ascending (left to right)
    decreasing.sort()
    # Sort increasing indices descending (right to left)
    increasing.sort(reverse=True)
    
    # The order of changes
    order = decreasing + increasing
    
    print(m)
    current_s = list(s)
    for idx in order:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()