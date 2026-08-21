import sys
from itertools import groupby

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # If strings are identical, it's always possible (0 operations)
    if s == t:
        print("Yes")
        return

    # For K=1, the edit distance must be <= 1.
    # We can check this by finding the first index where they differ.
    # We use a generator to find the first mismatch.
    diff_idx = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))

    # Case 1: Replace one character
    # Lengths must be equal, and only one character should differ.
    # We check if the suffixes after the first mismatch are identical 
    # if we skip the mismatched character in both.
    res_replace = (len(s) == len(t) and s[diff_idx + 1:] == t[diff_idx + 1:])

    # Case 2: Delete one character from S
    # S must be 1 character longer than T.
    # We check if skipping the mismatched character in S makes it equal to T.
    res_delete = (len(s) == len(t) + 1 and s[diff_idx + 1:] == t[diff_idx:])

    # Case 3: Insert one character into S (Delete one from T)
    # T must be 1 character longer than S.
    # We check if skipping the mismatched character in T makes it equal to S.
    res_insert = (len(t) == len(s) + 1 and s[diff_idx:] == t[diff_idx + 1:])

    if res_replace or res_delete or res_insert:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()