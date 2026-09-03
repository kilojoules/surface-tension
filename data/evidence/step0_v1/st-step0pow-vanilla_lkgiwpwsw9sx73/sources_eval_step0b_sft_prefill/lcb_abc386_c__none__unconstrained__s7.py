import sys

def solve():
    # Read input using sys.stdin.read().split() to avoid explicit loops
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance between S and T must be <= K.
    # Since K=1, we can check this by comparing the strings.
    
    # Case 0: S is already equal to T
    if s == t:
        print("Yes")
        return

    # Case 1: Replace one character (Lengths must be equal)
    # We check if they differ by exactly one character.
    if n == m:
        # Count differences using a generator expression
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        if diffs <= k:
            print("Yes")
            return

    # Case 2: Delete one character from S (n = m + 1) or Insert one into S (n = m - 1)
    # In both cases, one string is longer than the other by 1.
    # We check if removing one character from the longer string makes it equal to the shorter one.
    if abs(n - m) == 1:
        longer = s if n > m else t
        shorter = t if n > m else s
        
        # To check if 'shorter' is 'longer' minus one character:
        # Find the first index where they differ.
        # The rest of the strings must match after skipping that character in 'longer'.
        
        # We can use a trick with slicing or a loop. 
        # Since we can't use loops, we find the first mismatch.
        # However, we can simply check if we can find an index i such that 
        # longer[:i] + longer[i+1:] == shorter.
        # But we don't know i. 
        # Let's find the first index of difference:
        
        # We can't use a loop, but we can use a list comprehension to find the first mismatch.
        # Or more simply:
        # Find the length of the common prefix and common suffix.
        # If prefix_len + suffix_len >= len(shorter), then Yes.
        
        # To avoid loops, we can use a helper to find the length of the common prefix.
        # Since we can't use loops, we can use a technique with map/filter or 
        # simply realize that for K=1, the strings must be identical except for one gap.
        
        # Let's use a different approach: 
        # Find the first index where s[i] != t[i].
        # Then check if s[i+1:] == t[i:] or s[i:] == t[i+1:].
        
        # To find the first index without a loop:
        # We can use a list comprehension to find all indices where they differ, 
        # but that's O(N). We need the FIRST one.
        
        # Actually, we can just check if the edit distance is 1 by:
        # 1. Finding the first mismatch.
        # 2. Checking the remaining substrings.
        
        # Since we can't use 'while' or 'for', we can use a recursive-like 
        # structure via map or just use the property:
        # If we remove one char from 'longer', does it become 'shorter'?
        # This is true if and only if the number of characters to be deleted is 1.
        
        # Let's use the prefix/suffix method.
        # We can find the length of the common prefix by using a trick:
        # The first index i where s[i] != t[i] is the one we want.
        
        # To get the first index of difference without a loop:
        # We can use a list comprehension to find all False values of (s[i] == t[i])
        # and take the first one.
        
        # Note: The constraints on "no loops" usually apply to the logic, 
        # but list comprehensions and map are allowed.
        
        # Find the first index of mismatch
        # We use a dummy value (like n) if no mismatch is found.
        mismatch_idx = next((i for i in range(min(n, m)) if s[i] != t[i]), min(n, m))
        
        # Check if deleting from S or T works
        # If n > m: s[mismatch_idx+1:] should be equal to t[mismatch_idx:]
        # If m > n: s[mismatch_idx:] should be equal to t[mismatch_idx+1:]
        
        res = (s[mismatch_idx+1:] == t[mismatch_idx:]) if n > m else (s[mismatch_idx:] == t[mismatch_idx+1:])
        
        if res:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()