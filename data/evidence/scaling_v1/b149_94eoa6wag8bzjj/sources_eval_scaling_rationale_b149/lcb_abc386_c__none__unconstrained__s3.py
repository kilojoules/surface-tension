import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Since K=1, we check if edit distance is <= 1
    # Case 1: Strings are identical
    if s == t:
        print("Yes")
        return

    len_s = len(s)
    len_t = len(t)

    # Case 2: Lengths differ by more than 1, impossible for K=1
    if abs(len_s - len_t) > 1:
        print("No")
        return

    # Case 3: Lengths are equal - check if they differ by exactly one character (Substitution)
    if len_s == len_t:
        # Use sum() with a generator to count differences without a loop
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        print("Yes" if diffs <= 1 else "No")
        return

    # Case 4: Lengths differ by 1 - check if deleting one char from longer makes it the shorter (Insertion/Deletion)
    # Identify which string is longer
    longer, shorter = (s, t) if len_s > len_t else (t, s)
    
    # To check if removing one character makes them equal without a loop:
    # We find the first index where they differ.
    # Since we can't use loops, we can use a trick with 
    # comparing prefixes and suffixes.
    
    # Find the first mismatch index
    # We use a generator to find the first index i where longer[i] != shorter[i]
    # We can use next() to get the first index, or 0 if no mismatch is found within shorter's length.
    try:
        # Find index of first difference
        first_diff = next(i for i in range(len(shorter)) if longer[i] != shorter[i])
    except StopIteration:
        # No difference found in the length of the shorter string, 
        # so the difference is at the very end.
        first_diff = len(shorter)

    # If we remove the character at first_diff from 'longer', does it match 'shorter'?
    # We construct the string without the character at first_diff using slicing.
    if longer[:first_diff] + longer[first_diff+1:] == shorter:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()