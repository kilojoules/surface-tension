import sys

def solve():
    # Read K, S, and T from standard input
    # Using read().split() to handle potential whitespace/newline differences
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if the edit distance between S and T is <= K.
    # Since K=1, we can check this without a full DP table.
    
    # 1. If strings are identical, distance is 0.
    if s == t:
        print("Yes")
        return

    # 2. If length difference > 1, distance must be > 1.
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper function to check if two strings are 1 edit apart
    # We use a generator/comprehension to find the first index where characters differ.
    def check_one_edit(s1, s2):
        # Find the first index where s1 and s2 differ
        # zip stops at the shortest string length
        diff_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        # If no difference found in the overlapping prefix:
        if diff_idx is None:
            # They are identical up to the length of the shorter string.
            # They are 1 edit apart if the length difference is exactly 1.
            return abs(len(s1) - len(s2)) == 1
        
        # If a difference is found at diff_idx:
        # Case 1: Replace (lengths must be equal)
        # Check if the rest of the strings after the mismatch are identical.
        if len(s1) == len(s2):
            return s1[diff_idx + 1:] == s2[diff_idx + 1:]
        
        # Case 2: Insertion/Deletion
        # Identify which string is longer
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        # Check if skipping the mismatched character in the longer string 
        # makes it identical to the shorter string from that point onwards.
        return longer[diff_idx + 1:] == shorter[diff_idx:]

    if check_one_edit(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()