import sys
from functools import reduce

def solve():
    # Read input and handle potential empty lines/whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K. Given K=1, we can optimize.
    
    # 1. Quick length check: if length difference > 1, distance is definitely > 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # 2. Check if strings are identical
    if s == t:
        print("Yes")
        return

    # 3. Check if distance is exactly 1
    # We find the first index where characters differ
    # We use a generator to find the first mismatch index
    mismatch_idx = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))

    # Case A: Replace (lengths are equal)
    # s[i] != t[i], but suffixes after i must be identical
    if len(s) == len(t):
        if s[mismatch_idx + 1:] == t[mismatch_idx + 1:]:
            print("Yes")
            return

    # Case B: Delete from S (S is longer by 1)
    # s[i] is deleted, so s[i+1:] must match t[i:]
    if len(s) == len(t) + 1:
        if s[mismatch_idx + 1:] == t[mismatch_idx:]:
            print("Yes")
            return

    # Case C: Insert into S (T is longer by 1)
    # t[i] is inserted, so s[i:] must match t[i+1:]
    if len(t) == len(s) + 1:
        if s[mismatch_idx:] == t[mismatch_idx + 1:]:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()