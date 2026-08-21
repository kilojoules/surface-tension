import sys
from functools import reduce

def solve():
    # Read input and handle potential whitespace/newlines
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # The problem asks if the edit distance between S and T is <= K.
    # Since K=1, we can check this efficiently without a full DP table.
    
    # 1. Check if they are already identical
    if S == T:
        print("Yes")
        return

    # 2. Check if length difference is > 1
    if abs(len(S) - len(T)) > 1:
        print("No")
        return

    # Helper to check if S and T are 1 edit apart
    # We find the first index where characters differ
    def check_one_edit(s1, s2):
        # Find first mismatch
        # Using a generator to find the first index i where s1[i] != s2[i]
        mismatch_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        # If no mismatch found in the shorter length, they differ only by trailing chars
        if mismatch_idx is None:
            return abs(len(s1) - len(s2)) <= 1

        # Try the three possible operations at the mismatch point:
        # 1. Replace: check if s1[i+1:] == s2[i+1:]
        # 2. Delete from s1: check if s1[i+1:] == s2[i:]
        # 3. Insert into s1 (Delete from s2): check if s1[i:] == s2[i+1:]
        
        # We use slicing which is efficient in Python
        res = (
            s1[mismatch_idx + 1:] == s2[mismatch_idx + 1:] or # Replace
            s1[mismatch_idx + 1:] == s2[mismatch_idx:] or     # Delete from S
            s1[mismatch_idx:] == s2[mismatch_idx + 1:]        # Insert into S
        )
        return res

    if check_one_edit(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()