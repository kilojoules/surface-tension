import sys
from functools import reduce

def solve():
    # Read input and strip whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K.
    # Since K=1, we can check this without a full DP table.
    
    # 1. Length difference must be <= 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # 2. If lengths are equal, they must differ by at most 1 character (Substitution)
    if len(s) == len(t):
        diffs = reduce(lambda acc, pair: acc + (1 if pair[0] != pair[1] else 0), zip(s, t), 0)
        print("Yes" if diffs <= 1 else "No")
        return

    # 3. If lengths differ by 1, one must be obtainable by deleting 1 char from the longer one
    # Let 'longer' be the longer string and 'shorter' be the shorter one.
    longer, shorter = (s, t) if len(s) > len(t) else (t, s)
    
    # Find the first mismatch
    # We use a generator to find the first index where characters differ
    mismatch_idx = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
    
    # If we skip the mismatched character in the longer string, 
    # the remainder of the strings must be identical.
    # longer[mismatch_idx + 1:] == shorter[mismatch_idx:]
    if longer[mismatch_idx + 1:] == shorter[mismatch_idx:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()