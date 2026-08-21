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
    
    # The edit distance between two strings is the minimum number of operations
    # to transform one into the other. We need to check if edit_distance(s, t) <= k.
    # Since k=1, we can check this without a full DP table.
    
    # Case 0: Strings are identical
    if s == t:
        print("Yes")
        return

    # Case 1: Difference in length is more than 1
    if abs(n - m) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after removing one character
    # This handles both Insertion and Deletion cases.
    def check_one_diff(s1, s2):
        # s1 is the longer string, s2 is the shorter string
        # We check if removing one char from s1 makes it s2
        # Using slicing to avoid loops
        # We find the first point of difference
        # Since we can't use loops, we use a trick with zip and slicing
        
        # Find the index of the first mismatch
        # We use a list comprehension to find indices of mismatches
        mismatches = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
        
        if not mismatches:
            # s2 is a prefix of s1
            return True
        
        idx = mismatches[0]
        # Check if removing s1[idx] makes the rest match
        return s1[idx+1:] == s2[idx:]

    # Case 2: Length difference is 1 (Insertion or Deletion)
    if n == m + 1:
        # Try deleting one from S
        if check_one_diff(s, t):
            print("Yes")
            return
    elif m == n + 1:
        # Try deleting one from T (equivalent to inserting into S)
        if check_one_diff(t, s):
            print("Yes")
            return
    
    # Case 3: Lengths are equal (Replacement)
    elif n == m:
        # Count mismatches
        mismatches = [i for i, (a, b) in enumerate(zip(s, t)) if a != b]
        if len(mismatches) == 1:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()