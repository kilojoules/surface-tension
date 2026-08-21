import sys

def solve():
    # Read input and split into components
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the edit distance between S and T is <= K.
    # Given K=1, we can check this without a full DP table.
    
    # 1. If strings are identical, distance is 0.
    if s == t:
        print("Yes")
        return

    # 2. If length difference is > 1, distance must be > 1.
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are 1 edit apart
    # We use a generator to find the first mismatch
    def check_one_edit(s1, s2):
        # Find the first index where characters differ
        # Using a generator expression with next() to find the first mismatch
        mismatch_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        # If no mismatch found in the shorter string's length
        if mismatch_idx is None:
            # One string is a prefix of the other; check if length difference is exactly 1
            return abs(len(s1) - len(s2)) == 1
        
        # If mismatch found at index i:
        # Case 1: Replace (s1[i] -> s2[i]), then remaining suffixes must be identical
        # Case 2: Delete from s1 (s1[i+1:] == s2[i:])
        # Case 3: Insert into s1 (s1[i:] == s2[i+1:])
        
        # We use slicing to check the remaining parts of the strings
        return (s1[mismatch_idx + 1:] == s2[mismatch_idx + 1:]) or \
               (s1[mismatch_idx + 1:] == s2[mismatch_idx:]) or \
               (s1[mismatch_idx:] == s2[mismatch_idx + 1:])

    if check_one_edit(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()