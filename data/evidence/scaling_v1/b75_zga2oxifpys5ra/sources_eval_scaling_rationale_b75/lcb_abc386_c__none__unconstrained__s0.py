import sys

def solve():
    # Read input and handle potential whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Case 0: Strings are already identical
    if S == T:
        print("Yes")
        return

    # Since K=1, we only need to check if one operation suffices.
    # We check the three possible operations:
    
    # 1. Replacement: Lengths must be equal. 
    # There must be exactly one position where characters differ.
    def check_replace():
        return (len(S) == len(T) and 
                sum(1 for s, t in zip(S, T) if s != t) <= 1)

    # 2. Deletion from S (or Insertion into T): len(S) = len(T) + 1
    # We look for a split point i such that S[:i] + S[i+1:] == T
    def check_delete():
        if len(S) != len(T) + 1:
            return False
        # Find the first mismatch
        # We use a generator to find the first index where S and T differ
        # Since we can't use loops, we can't easily find the index.
        # However, we can check if there exists an i such that S[:i] + S[i+1:] == T.
        # But iterating i from 0 to len(S) is a loop.
        # Instead, we find the first mismatch index using a trick with zip and next.
        mismatch_idx = next((i for i, (s, t) in enumerate(zip(S, T)) if s != t), len(T))
        return S[:mismatch_idx] + S[mismatch_idx+1:] == T

    # 3. Insertion into S (or Deletion from T): len(T) = len(S) + 1
    def check_insert():
        if len(T) != len(S) + 1:
            return False
        mismatch_idx = next((i for i, (s, t) in enumerate(zip(S, T)) if s != t), len(S))
        return S[:mismatch_idx] + T[mismatch_idx+1:] == S # This is wrong logic
        # Correct logic for insertion: S[:i] + T[i+1:] == S is wrong.
        # It should be: T[:mismatch_idx] + T[mismatch_idx+1:] == S
        # Let's redefine:
    
    # Corrected helper for Insertion/Deletion logic:
    # If we remove one char from the longer string, does it become the shorter string?
    def is_one_diff(longer, shorter):
        if len(longer) != len(shorter) + 1:
            return False
        # Find first mismatch
        idx = next((i for i, (l, s) in enumerate(zip(longer, shorter)) if l != s), len(shorter))
        return longer[:idx] + longer[idx+1:] == shorter

    # Final check combining all possibilities
    if (check_replace() or 
        is_one_diff(S, T) or 
        is_one_diff(T, S)):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()