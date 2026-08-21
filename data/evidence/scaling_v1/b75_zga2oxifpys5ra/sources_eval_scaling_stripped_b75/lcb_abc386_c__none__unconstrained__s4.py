import sys
from functools import reduce

def solve():
    # Read input and handle potential empty lines or trailing spaces
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K.
    # Given K=1 and string lengths up to 500,000, a full DP table is O(N^2).
    # However, we only care if the distance is 0 or 1.
    
    # Case 0: Strings are identical
    if s == t:
        print("Yes")
        return

    # If K=0 and strings aren't identical, it's impossible.
    # But the prompt says K=1 for this sub-problem.
    if k == 0:
        print("No")
        return

    # For K=1, the distance is 1 if:
    # 1. One replacement: lengths are equal, and they differ by exactly one character.
    # 2. One deletion: len(s) == len(t) + 1, and t is a subsequence of s.
    # 3. One insertion: len(t) == len(s) + 1, and s is a subsequence of t.

    def is_one_edit_distance(s1, s2):
        n, m = len(s1), len(s2)
        if abs(n - m) > 1:
            return False
        
        # Ensure s1 is the shorter string
        if n > m:
            s1, s2 = s2, s1
            n, m = m, n
        
        # Find the first mismatch
        # We use a generator to find the first index where characters differ
        mismatch = next((i for i in range(n) if s1[i] != s2[i]), n)
        
        if mismatch == n:
            # s1 is a prefix of s2, check if length difference is <= 1
            return m - n <= 1
        
        # If mismatch found, check the remaining suffixes based on operation
        if n == m:
            # Replace: check if the rest of the strings are identical
            return s1[mismatch+1:] == s2[mismatch+1:]
        else:
            # Insert into s1 (Delete from s2): check if s1 from mismatch onwards 
            # matches s2 from mismatch + 1 onwards
            return s1[mismatch:] == s2[mismatch+1:]

    if is_one_edit_distance(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()