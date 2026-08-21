import sys
from functools import reduce

def solve():
    # Read input and handle potential empty lines or trailing spaces
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K.
    # Given K=1, we can check this in O(N) time.
    
    n, m = len(s), len(t)
    
    # If length difference is more than 1, edit distance is definitely > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Find the first position where s and t differ
    # We use a generator to find the first index i where s[i] != t[i]
    # We use a helper to find the mismatch index safely
    def get_first_diff(s1, s2):
        # Use a list comprehension to find all indices where they differ
        # and take the first one.
        diffs = [i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]]
        return diffs[0] if diffs else min(len(s1), len(s2))

    first_diff = get_first_diff(s, t)

    # If no difference was found within the length of the shorter string,
    # the distance is simply the difference in lengths.
    if first_diff == min(n, m):
        print("Yes" if abs(n - m) <= k else "No")
        return

    # If a difference is found at index i, we have 3 possibilities to check:
    # 1. Replace: s[i] becomes t[i]. Check if s[i+1:] == t[i+1:]
    # 2. Delete: s[i] is removed. Check if s[i+1:] == t[i:]
    # 3. Insert: t[i] is inserted into s. Check if s[i:] == t[i+1:]
    
    # We use a list of booleans and 'any()' to check if any of these work.
    possibilities = [
        s[first_diff + 1:] == t[first_diff + 1:], # Replace
        s[first_diff + 1:] == t[first_diff:],     # Delete from S
        s[first_diff:] == t[first_diff + 1:]      # Insert into S
    ]

    if any(possibilities):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()