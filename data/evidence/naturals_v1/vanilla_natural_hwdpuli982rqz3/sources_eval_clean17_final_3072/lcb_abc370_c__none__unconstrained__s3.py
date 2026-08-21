import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # To minimize M, we must change only characters that differ between S and T.
    # M is the number of indices i where S[i] != T[i].
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To make the array X lexicographically smallest:
    # We want X_1 to be as small as possible, then X_2, and so on.
    # A string is smaller if its first differing character is smaller.
    # This means we should prioritize changing characters at smaller indices first
    # IF the target character T[i] is smaller than the current character S[i].
    # If T[i] is larger than S[i], changing it immediately makes the string larger.
    # In that case, we should delay that change as long as possible to keep 
    # the prefix smaller for longer.
    
    # Correct Strategy for Lexicographical Minimum:
    # 1. Identify indices where S[i] != T[i].
    # 2. Divide these into two groups:
    #    Group A: Indices i where T[i] < S[i] (Changing these makes the string smaller)
    #    Group B: Indices i where T[i] > S[i] (Changing these makes the string larger)
    # 3. Process Group A in increasing order of index (to make the prefix smaller as early as possible).
    # 4. Process Group B in decreasing order of index (to keep the prefix smaller for as long as possible).
    
    group_a = [i for i in diff_indices if T[i] < S[i]]
    group_b = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort group_a ascending, group_b descending
    group_a.sort()
    group_b.sort(reverse=True)
    
    # The sequence of indices to change
    change_order = group_a + group_b
    
    # Generate the array X
    # We use a list of characters because strings are immutable in Python
    current_S = list(S)
    X = []
    
    for idx in change_order:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output results
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()