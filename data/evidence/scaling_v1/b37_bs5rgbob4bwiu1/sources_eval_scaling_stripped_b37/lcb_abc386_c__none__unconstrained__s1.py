import sys

def solve():
    # Read input and handle potential whitespace/newline issues
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is always 1 for this specific sub-problem
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem asks if the edit distance between S and T is <= K.
    # Since K=1, we can check this without a full DP table.
    
    # 1. Check if they are already identical
    if s == t:
        print("Yes")
        return

    # 2. Check if length difference is greater than 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper function to check if S and T are 1 edit apart
    # We use a generator/comprehension to find the first index where they differ
    def check_one_edit(s1, s2):
        # Find the first index where characters differ
        # Using next() with a default value to avoid StopIteration
        diff_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        # If no difference found in the shortest length, 
        # they are 1 edit apart if length difference is exactly 1
        if diff_idx is None:
            return abs(len(s1) - len(s2)) <= 1

        # If a difference is found at diff_idx:
        # Case 1: Replace (s1[diff_idx] -> s2[diff_idx])
        # Check if the suffixes after the mismatch are identical
        # Case 2: Delete from s1 (s1[diff_idx] is removed)
        # Check if s1[diff_idx+1:] == s2[diff_idx:]
        # Case 3: Insert into s1 (s2[diff_idx] is inserted)
        # Check if s1[diff_idx:] == s2[diff_idx+1:]
        
        # We use slicing which is efficient in Python
        # Replace
        if s1[diff_idx+1:] == s2[diff_idx+1:]:
            return True
        # Delete
        if s1[diff_idx+1:] == s2[diff_idx:]:
            return True
        # Insert
        if s1[diff_idx:] == s2[diff_idx+1:]:
            return True
            
        return False

    if check_one_edit(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()