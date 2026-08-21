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

    # The problem is to check if the edit distance between S and T is <= 1.
    # Since K=1, we can check this by comparing the strings.
    
    # 1. Identical strings
    if s == t:
        print("Yes")
        return

    # 2. Length difference must be at most 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after removing one character
    # We use groupby to find the first index where they differ
    # zip(s, t) allows us to find the first mismatch
    
    # Find the first index where s and t differ
    # We use a generator to find the first index i where s[i] != t[i]
    # Since we can't use loops, we use next() with a generator expression
    try:
        # Find first mismatch
        first_diff = next(i for i, (a, b) in enumerate(zip(s, t)) if a != b)
    except StopIteration:
        # One string is a prefix of the other, and length difference is <= 1
        # This is already covered by the length check and the fact they aren't identical
        print("Yes")
        return

    # Check the three possible edit operations:
    # Replace: s[first_diff+1:] == t[first_diff+1:] (and lengths are equal)
    # Delete from S: s[first_diff+1:] == t[first_diff:] (len(s) == len(t) + 1)
    # Insert into S: s[first_diff:] == t[first_diff+1:] (len(t) == len(s) + 1)
    
    # We evaluate these conditions in a list and use 'any()'
    results = [
        (len(s) == len(t) and s[first_diff + 1:] == t[first_diff + 1:]),
        (len(s) == len(t) + 1 and s[first_diff + 1:] == t[first_diff:]),
        (len(t) == len(s) + 1 and s[first_diff:] == t[first_diff + 1:])
    ]

    if any(results):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()