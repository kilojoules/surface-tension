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
        if s == t:
            print(0)
            return
        else:
            return

    n = len(s)
    # Identify positions where S and T differ
    diff_indices = [i for i in range(n) if s[i] != t[i]]
    m = len(diff_indices)
    
    if m == 0:
        print(0)
        return

    # To make the array of strings X lexicographically smallest:
    # We want X_1 to be as small as possible, then X_2, and so on.
    # A string is smaller if its first differing character is smaller.
    # Thus, we should prioritize changing characters at the earliest possible 
    # positions in the string to the target characters, BUT only if 
    # the target character is smaller than the current character.
    # If the target character is larger, we should delay that change as much 
    # as possible to keep the string lexicographically smaller for longer.

    # Strategy:
    # 1. First, change all characters where T[i] < S[i], processed from left to right.
    # 2. Then, change all characters where T[i] > S[i], processed from right to left.
    
    # This ensures that the strings added to X start with the smallest possible 
    # characters as early as possible.
    
    to_change_smaller = [] # indices where T[i] < S[i]
    to_change_larger = []   # indices where T[i] > S[i]
    
    for i in diff_indices:
        if t[i] < s[i]:
            to_change_smaller.append(i)
        else:
            to_change_larger.append(i)
            
    # Sort smaller changes ascending (left to right)
    to_change_smaller.sort()
    # Sort larger changes descending (right to left)
    to_change_larger.sort(reverse=True)
    
    order_of_change = to_change_smaller + to_change_larger
    
    print(m)
    current_s = list(s)
    for idx in order_of_change:
        current_s[idx] = t[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()