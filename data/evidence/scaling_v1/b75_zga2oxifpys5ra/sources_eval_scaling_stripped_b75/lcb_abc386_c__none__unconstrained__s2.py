import sys

def solve():
    # Read input and split into components
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K. Given K=1, we can check this in O(N) time.
    
    # 1. If strings are identical, distance is 0
    if s == t:
        print("Yes")
        return

    # 2. If length difference is more than 1, distance must be > 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after ignoring one character
    # This covers Insertion and Deletion
    def check_one_diff(s1, s2):
        # Ensure s1 is the longer string
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        # Try removing one character from the longer string at all possible positions
        # However, a naive loop is O(N^2). We use a two-pointer approach.
        i = 0
        while i < len(s2) and s1[i] == s2[i]:
            i += 1
        # After the first mismatch, the rest of s2 must match s1 from the next character
        return s1[i+1:] == s2[i:]

    # Helper to check if strings are equal after one replacement
    def check_replacement(s1, s2):
        if len(s1) != len(s2):
            return False
        # Count positions where characters differ
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        return len(diffs) == 1

    # Check all three operation types for K=1
    if check_one_diff(s, t) or check_replacement(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()