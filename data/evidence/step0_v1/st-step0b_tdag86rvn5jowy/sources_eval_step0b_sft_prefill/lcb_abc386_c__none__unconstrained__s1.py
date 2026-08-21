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
    # Given K=1, we can check this without a full DP table.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Length difference > 1 means Edit Distance > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after removing one character
    # This handles both Insertion and Deletion cases.
    def is_one_edit_apart(s1, s2):
        # Ensure s1 is the shorter string
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        # If lengths are same, check for one replacement
        if len(s1) == len(s2):
            diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
            return len(diffs) <= 1
        
        # If lengths differ by 1, check for one insertion/deletion
        # Find the first mismatch
        first_diff = 0
        while first_diff < len(s1) and s1[first_diff] == s2[first_diff]:
            first_diff += 1
        
        # The rest of s1 must match s2 starting from the next character
        return s1[first_diff:] == s2[first_diff + 1:]

    if is_one_edit_apart(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()