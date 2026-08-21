import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # The problem is to check if the edit distance between S and T is <= K.
    # Since K=1, we can check this without a full DP table.
    
    # 1. If strings are identical, distance is 0.
    if S == T:
        print("Yes")
        return

    # 2. If length difference is more than 1, distance is definitely > 1.
    if abs(len(S) - len(T)) > 1:
        print("No")
        return

    # Helper to check if two strings are 1 edit apart
    # We use a generator to find the first index where characters differ.
    def check_one_edit(s1, s2):
        # Find the first mismatch
        # Use next() with a default value to avoid StopIteration
        mismatch_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        # If no mismatch found in the shorter length, they differ only by trailing chars
        if mismatch_idx is None:
            return abs(len(s1) - len(s2)) <= 1

        # If mismatch found at i, we have three possibilities:
        # 1. Replace: s1[i+1:] == s2[i+1:]
        # 2. Delete from s1: s1[i+1:] == s2[i:]
        # 3. Insert into s1: s1[i:] == s2[i+1:]
        
        # We check these based on the lengths of the strings
        # Replace
        if len(s1) == len(s2):
            return s1[mismatch_idx + 1:] == s2[mismatch_idx + 1:]
        # Delete from s1 (s1 is longer)
        if len(s1) > len(s2):
            return s1[mismatch_idx + 1:] == s2[mismatch_idx:]
        # Insert into s1 (s2 is longer)
        if len(s1) < len(s2):
            return s1[mismatch_idx:] == s2[mismatch_idx + 1:]
        
        return False

    if check_one_edit(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()