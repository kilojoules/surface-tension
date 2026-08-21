import sys

def solve():
    # Read input from standard input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        k = int(line1.strip())
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except ValueError:
        return

    # The problem asks if the edit distance between S and T is <= K.
    # For K=1, the edit distance is <= 1 if:
    # 1. S == T (distance 0)
    # 2. S can be transformed to T by one replacement, insertion, or deletion.

    if s == t:
        print("Yes")
        return

    # If the difference in lengths is more than 1, edit distance cannot be 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    n, m = len(s), len(t)

    # Case 1: Replacement (lengths are equal)
    if n == m:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > 1:
                print("No")
                return
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Insertion or Deletion (length difference is exactly 1)
    # Let 'short' be the shorter string and 'long' be the longer string.
    if n < m:
        short, long = s, t
    else:
        short, long = t, s
    
    # We need to check if 'short' is a subsequence of 'long' 
    # and can be obtained by deleting exactly one character from 'long'.
    # This is equivalent to checking if we can match short into long by skipping one char.
    i = 0 # index for short
    j = 0 # index for long
    diffs = 0
    
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            j += 1 # skip the character in the longer string
            if diffs > 1:
                print("No")
                return
    
    # If we reached the end of short, we check if we used at most 1 skip.
    # The loop finishes when i reaches len(short) or j reaches len(long).
    # If j hasn't reached the end, the remaining characters in long are also 'diffs'.
    diffs += (len(long) - j)
    
    if diffs <= 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()