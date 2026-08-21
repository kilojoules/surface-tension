import sys

def solve():
    # Read S and T from standard input
    try:
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
    except EOFError:
        return

    if not S or not T:
        # Handle potential empty input cases
        if S == T:
            print(0)
            return
        # If one is empty and the other isn't, constraints say equal length, 
        # but we handle for safety.
        return

    n = len(S)
    diff_indices = []
    for i in range(n):
        if S[i] != T[i]:
            diff_indices.append(i)

    # The minimum number of elements M is the number of characters that differ.
    m = len(diff_indices)
    print(m)

    # To get the lexicographically smallest array X:
    # We want X_1 to be as small as possible, then X_2, and so on.
    # A string is smaller if its first differing character is smaller.
    # Thus, we should prioritize changing characters at the earliest possible indices 
    # that would make the string lexicographically smaller.
    
    # Specifically, if S[i] > T[i], changing S[i] to T[i] makes the string smaller.
    # If S[i] < T[i], changing S[i] to T[i] makes the string larger.
    
    # Strategy:
    # 1. Process all indices i where S[i] > T[i] from left to right.
    # 2. Process all indices i where S[i] < T[i] from right to left.
    
    # This ensures that we reduce the string's value as quickly as possible 
    # and delay increasing it as long as possible.
    
    decrease_indices = []
    increase_indices = []
    
    for i in diff_indices:
        if S[i] > T[i]:
            decrease_indices.append(i)
        else:
            increase_indices.append(i)
            
    # Sort decrease indices ascending (left to right)
    decrease_indices.sort()
    # Sort increase indices descending (right to left)
    increase_indices.sort(reverse=True)
    
    order = decrease_indices + increase_indices
    
    current_s = list(S)
    for idx in order:
        current_s[idx] = T[idx]
        print("".join(current_s))

if __name__ == "__main__":
    solve()