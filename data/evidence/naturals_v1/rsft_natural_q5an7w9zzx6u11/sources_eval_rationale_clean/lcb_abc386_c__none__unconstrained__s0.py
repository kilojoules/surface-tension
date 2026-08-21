import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # For K=1, we check if the edit distance is 0 or 1.
    # Edit distance 0: strings are identical.
    # Edit distance 1: 
    # 1. Replace: lengths are equal, differ by exactly one character.
    # 2. Delete: len(s) == len(t) + 1, s becomes t by removing one char.
    # 3. Insert: len(t) == len(s) + 1, t becomes s by removing one char.

    # Helper to check if s can become t by removing one character
    # We find the first mismatch and check if the rest of the strings match.
    # Since we can't use loops, we use a trick with zip and next to find the first mismatch index.
    
    def check_one_diff(s1, s2):
        # This checks if s1 can be transformed to s2 by replacing one character
        # Only called when len(s1) == len(s2)
        diffs = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
        return len(diffs) <= 1

    def check_one_removal(longer, shorter):
        # This checks if 'longer' can become 'shorter' by removing one character
        # We find the first index where they differ
        # We use a generator to find the first mismatch index
        mismatch_idx = next((i for i, (a, b) in enumerate(zip(longer, shorter)) if a != b), len(shorter))
        # Check if removing the character at mismatch_idx makes them equal
        return longer[:mismatch_idx] + longer[mismatch_idx+1:] == shorter

    # The conditions for Yes:
    # 1. S == T
    # 2. len(S) == len(T) and they differ by 1 char
    # 3. len(S) == len(T) + 1 and removing 1 char from S gives T
    # 4. len(T) == len(S) + 1 and removing 1 char from T gives S
    
    results = [
        s == t,
        (len(s) == len(t) and check_one_diff(s, t)),
        (len(s) == len(t) + 1 and check_one_removal(s, t)),
        (len(t) == len(s) + 1 and check_one_removal(t, s))
    ]

    if any(results):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()