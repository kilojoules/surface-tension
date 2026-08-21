import sys

def solve():
    # Read K, S, and T from standard input
    # Using read().split() to handle all whitespace variations
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if the edit distance between S and T is <= K.
    # Given K=1, we can check this efficiently without a full DP table.
    
    # 1. If strings are identical, distance is 0.
    if s == t:
        print("Yes")
        return

    # 2. If length difference > 1, distance is definitely > 1.
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are 1 edit apart
    def is_one_edit(s1, s2):
        n, m = len(s1), len(s2)
        # Ensure s1 is the shorter string
        if n > m:
            s1, s2 = s2, s1
            n, m = m, n
        
        # Case 1: Replace (lengths are equal)
        if n == m:
            # Count positions where characters differ
            diffs = [i for i in range(n) if s1[i] != s2[i]]
            return len(diffs) == 1
        
        # Case 2: Insert/Delete (length difference is exactly 1)
        # Find the first mismatch
        for i in range(n):
            if s1[i] != s2[i]:
                # If we skip the mismatch in the longer string, 
                # the rest must be identical.
                return s1[i:] == s2[i+1:]
        
        # If no mismatch found in the first n characters, 
        # the difference is just the last character of the longer string.
        return True

    if is_one_edit(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()