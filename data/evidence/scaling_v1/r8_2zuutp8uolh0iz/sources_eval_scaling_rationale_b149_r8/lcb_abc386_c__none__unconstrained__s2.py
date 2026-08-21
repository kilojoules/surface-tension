import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem specifies K=1. 
    # We need to check if edit_distance(s, t) <= 1.
    
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # If length difference is > 1, it's impossible with 1 operation
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if s and t are identical after removing one character 
    # at a specific index. Used for both insertion and deletion.
    # We use a generator expression and any() to avoid explicit loops.
    
    # Case 1: Lengths are equal (Possible Replace)
    # Check if they differ by exactly one character.
    if len(s) == len(t):
        # Count mismatches using a generator expression
        diffs = sum(1 for i in range(len(s)) if s[i] != t[i])
        if diffs <= 1:
            print("Yes")
            return

    # Case 2: Lengths differ by 1 (Possible Insert or Delete)
    # We identify the first point of divergence and check if the rest matches.
    # To avoid loops, we find the first mismatch index using a next() call.
    # We compare the strings by skipping one character in the longer string.
    
    # Identify which is longer
    longer, shorter = (s, t) if len(s) > len(t) else (t, s)
    
    # Find the first index where they differ
    # We use a generator to find the first index i where longer[i] != shorter[i]
    # If no mismatch is found within the length of 'shorter', the difference is at the end.
    first_mismatch = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
    
    # Check if removing the character at first_mismatch from 'longer' makes it equal to 'shorter'
    # We use slicing to construct the string without the character at first_mismatch.
    if longer[:first_mismatch] + longer[first_mismatch+1:] == shorter:
        print("Yes")
        return

    print("No")

if __name__ == "__main__":
    solve()