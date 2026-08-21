import sys
from functools import reduce

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To get the lexicographically smallest array X, we must consider:
    # 1. If T[i] < S[i], we want to change this character as early as possible
    #    to make the resulting string smaller.
    # 2. If T[i] > S[i], we want to change this character as late as possible
    #    to avoid making the string larger prematurely.
    
    # Sort indices: those where T[i] < S[i] come first (ascending),
    # then those where T[i] > S[i] come last (descending).
    # However, the simplest way to ensure lexicographical minimality is:
    # Process all i where T[i] < S[i] in increasing order of i,
    # then process all i where T[i] > S[i] in decreasing order of i.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing group ascending by index, and increasing group descending by index
    # Wait, the logic is: 
    # To make the string smallest, we want the leftmost characters to decrease first.
    # For characters that must increase, we want the rightmost ones to increase first
    # so the prefix stays small for as long as possible.
    
    order = sorted([i for i in diff_indices if T[i] < S[i]]) + \
            sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)

    # Use reduce to generate the sequence of strings.
    # The accumulator stores the list of strings X.
    def transform(acc, idx):
        current_s = acc[-1] if acc else S
        # Create new string by replacing character at idx
        new_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [new_s]

    X = reduce(transform, order, [])

    # Output the result
    sys.stdout.write(f"{len(X)}\n")
    if X:
        sys.stdout.write("\n".join(X) + "\n")

if __name__ == "__main__":
    solve()