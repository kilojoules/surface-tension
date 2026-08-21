import sys
from itertools import groupby

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # The problem asks if the edit distance between S and T is <= K.
    # Since K=1, we can check this by comparing the strings.
    # If S == T, distance is 0.
    # If length difference > 1, distance is > 1.
    
    # We use a helper to check if one string is the other with one character removed.
    # This covers both Insertion and Deletion.
    def is_one_diff(s1, s2):
        # Find the first index where they differ
        # We use a generator to find the first mismatch
        diff_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))
        
        # If we reached the end, they are identical up to the shorter length
        # Check if the remaining part of the longer string is just one character
        if diff_idx == min(len(s1), len(s2)):
            return abs(len(s1) - len(s2)) <= 1
        
        # If they differ at diff_idx:
        # 1. Try replacing: check if s1[diff_idx+1:] == s2[diff_idx+1:]
        # 2. Try deleting from s1: check if s1[diff_idx+1:] == s2[diff_idx:]
        # 3. Try deleting from s2: check if s1[diff_idx:] == s2[diff_idx+1:]
        
        # Replace
        if len(s1) == len(s2) and s1[diff_idx+1:] == s2[diff_idx+1:]:
            return True
        # Delete from s1
        if len(s1) == len(s2) + 1 and s1[diff_idx+1:] == s2[diff_idx:]:
            return True
        # Delete from s2 (Insert into s1)
        if len(s2) == len(s1) + 1 and s1[diff_idx:] == s2[diff_idx+1:]:
            return True
            
        return False

    # Special case for K=0 (though constraints say K=1, the logic should hold)
    if K == 0:
        print("Yes" if S == T else "No")
        return

    # For K=1, we check if they are identical or if one edit suffices.
    if S == T or is_one_diff(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()