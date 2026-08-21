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

    # The problem is to check if Edit Distance (Levenshtein) <= 1
    # Since K=1, we can check this without a full DP table.
    
    # 1. Check if they are already identical
    if s == t:
        print("Yes")
        return

    # 2. Check if length difference is > 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are identical after removing one character
    # We use groupby to find the first index where they differ
    def check_one_diff(s1, s2):
        # Find the first index where s1 and s2 differ
        # We zip them and find the first pair that is different
        diff_idx = next((i for i, (a, b) in enumerate(zip(s1, s2)) if a != b), min(len(s1), len(s2)))
        
        # Case 1: Replace (lengths equal)
        if len(s1) == len(s2):
            # Check if the rest of the strings after the mismatch are identical
            return s1[diff_idx + 1:] == s2[diff_idx + 1:]
        
        # Case 2: Deletion from s1 / Insertion into s2
        if len(s1) > len(s2):
            # Try removing char at diff_idx from s1
            return s1[diff_idx + 1:] == s2[diff_idx:]
        
        # Case 3: Insertion into s1 / Deletion from s2
        if len(s1) < len(s2):
            # Try removing char at diff_idx from s2
            return s1[diff_idx:] == s2[diff_idx + 1:]
        
        return False

    if check_one_diff(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()