import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is guaranteed to be 1 for this sub-problem
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # If strings are already identical, 0 operations are needed
    if s == t:
        print("Yes")
        return

    # If the length difference is more than 1, it's impossible with K=1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Case 1: Replace one character (lengths must be equal)
    # We count how many characters differ at the same index
    if len(s) == len(t):
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        if diffs <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Insert or Delete one character
    # We identify the first point where the strings diverge
    # Then we check if the rest of the strings match after skipping one char
    
    # Ensure s is the shorter string for a unified "insertion" check
    # If len(s) > len(t), deleting from s is equivalent to inserting into t
    s_short, s_long = (s, t) if len(s) < len(t) else (t, s)
    
    # Find the first index where characters differ
    # We use a generator to find the first mismatch
    mismatch_idx = next((i for i in range(len(s_short)) if s_short[i] != s_long[i]), len(s_short))
    
    # Check if skipping the character in the longer string makes them identical
    # The remaining part of s_short from mismatch_idx should match 
    # the remaining part of s_long from mismatch_idx + 1
    if s_short[mismatch_idx:] == s_long[mismatch_idx + 1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()