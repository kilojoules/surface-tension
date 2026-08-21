import sys

def solve():
    # Read K, S, and T from standard input
    # Using read().split() to handle any whitespace variations
    input_data = syss = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem specifies K=1 for this sub-problem.
    # We need to check if the edit distance between S and T is <= 1.
    
    # 1. Check if they are already identical
    if s == t:
        print("Yes")
        return

    # 2. Check if length difference is greater than 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # 3. Case: Replace one character (lengths are equal)
    if len(s) == len(t):
        # Count positions where characters differ
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
        else:
            print("No")
        return

    # 4. Case: Insert or Delete one character (length difference is exactly 1)
    # Ensure 's' is the shorter string for a unified logic (insertion into s / deletion from t)
    if len(s) > len(t):
        s, t = t, s
    
    # Now len(t) == len(s) + 1
    # We look for the first index where s and t differ
    # s: abc
    # t: abxc
    # We can skip one char in t and check if the rest matches s
    
    # Find first mismatch
    # Using a generator with next() to find the first index i where s[i] != t[i]
    # If no mismatch is found in the range of s, the mismatch is at the very end.
    first_mismatch = next((i for i in range(len(s)) if s[i] != t[i]), len(s))
    
    # Check if s[first_mismatch:] == t[first_mismatch + 1:]
    if s[first_mismatch:] == t[first_mismatch + 1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()