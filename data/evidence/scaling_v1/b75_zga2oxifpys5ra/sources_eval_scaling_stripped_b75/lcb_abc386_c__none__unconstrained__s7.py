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

    # The problem is to check if the Levenshtein distance between S and T is <= K.
    # Since K=1, we can use a simplified approach.
    # If the length difference is > 1, it's impossible.
    if abs(len(s) - len(t)) > k:
        print("No")
        return

    # For K=1, we can check the 3 possible operations:
    # 1. Replace: lengths are equal, they differ by at most 1 character.
    # 2. Delete: len(s) = len(t) + 1, removing one char from s makes it t.
    # 3. Insert: len(t) = len(s) + 1, removing one char from t makes it s.
    
    def check_one_diff(s1, s2):
        # Checks if s1 and s2 differ by exactly one replacement
        if len(s1) != len(s2): return False
        diffs = [i for i in range(len(s1)) if s1[i] != s2[i]]
        return len(diffs) <= 1

    def check_one_deletion(longer, shorter):
        # Checks if removing one char from 'longer' makes it 'shorter'
        # Find first mismatch
        n, m = len(longer), len(shorter)
        # We can use a generator to find the first index where they differ
        first_diff = next((i for i in range(m) if longer[i] != shorter[i]), m)
        # Check if the rest of the strings match after skipping one char in 'longer'
        return longer[first_diff + 1:] == shorter[first_diff:]

    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replace (Length must be equal)
    if len(s) == len(t):
        if check_one_diff(s, t):
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Delete from S (S is longer)
    if len(s) == len(t) + 1:
        if check_one_deletion(s, t):
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Insert into S / Delete from T (T is longer)
    if len(t) == len(s) + 1:
        if check_one_deletion(t, s):
            print("Yes")
        else:
            print("No")
        return

    # Otherwise, length difference is too great
    print("No")

if __name__ == "__main__":
    solve()