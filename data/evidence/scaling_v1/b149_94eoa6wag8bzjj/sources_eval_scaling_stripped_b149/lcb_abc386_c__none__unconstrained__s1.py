import sys

def solve():
    # Read input and strip whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is given as 1 according to the problem statement
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # If length difference is more than 1, it's impossible with K=1
    if abs(n - m) > 1:
        print("No")
        return

    # If strings are identical, 0 operations needed
    if s == t:
        print("Yes")
        return

    # Helper to check if s can become t by deleting one char from s
    # (Used for both deletion from S and insertion into S)
    def check_one_diff(s1, s2):
        # s1 is the longer string, s2 is the shorter
        # We look for the first mismatch
        for i in range(len(s2)):
            if s1[i] != s2[i]:
                # Try skipping the character in s1 and check if the rest matches
                return s1[i+1:] == s2[i:]
        # If no mismatch found in s2's length, the extra char must be at the end
        return True

    # Case 1: Replace one character (lengths must be equal)
    if n == m:
        # Count positions where characters differ
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
        else:
            print("No")
            
    # Case 2: Delete one character from S (S is longer)
    elif n == m + 1:
        if check_one_diff(s, t):
            print("Yes")
        else:
            print("No")
            
    # Case 3: Insert one character into S (T is longer)
    elif m == n + 1:
        if check_one_diff(t, s):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()