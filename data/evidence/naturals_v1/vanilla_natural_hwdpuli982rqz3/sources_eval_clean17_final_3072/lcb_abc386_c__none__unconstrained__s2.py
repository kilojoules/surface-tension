import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The problem asks if the Edit Distance between S and T is <= K.
    # Given K = 1, we can check this without a full DP table.
    
    # Case 0: S and T are identical
    if s == t:
        print("Yes")
        return

    # Case 1: Difference in length is more than 1
    if abs(n - m) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after removing one character
    # This handles both Insertion (remove from longer) and Deletion (remove from shorter)
    def is_one_edit_away(s1, s2):
        # Ensure s1 is the shorter string
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        # Find the first mismatch
        # We use a list comprehension to find the index of the first difference
        # Since we can't use loops, we use a trick with map/next
        try:
            # Find index of first difference
            # We zip the strings and find the first index where they differ
            diff_idx = next(i for i, (a, b) in enumerate(zip(s1, s2)) if a != b)
        except StopIteration:
            # One string is a prefix of the other
            return True

        # If lengths are same, it's a replacement: check if the rest is identical
        if len(s1) == len(s2):
            return s1[diff_idx + 1:] == s2[diff_idx + 1:]
        
        # If lengths differ, it's an insertion/deletion: 
        # check if s1 from diff_idx matches s2 from diff_idx + 1
        return s1[diff_idx:] == s2[diff_idx + 1:]

    if is_one_edit_away(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()