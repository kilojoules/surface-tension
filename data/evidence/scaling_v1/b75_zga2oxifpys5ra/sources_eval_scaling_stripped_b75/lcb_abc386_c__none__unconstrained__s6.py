import sys

def solve():
    # Read input and split into tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is always 1 for this specific sub-problem
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # If strings are already identical, 0 operations needed
    if s == t:
        print("Yes")
        return

    # If the length difference is more than 1, it's impossible with K=1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Case 1: Replace (Lengths are equal)
    if len(s) == len(t):
        # Count positions where characters differ
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if len(diffs) <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Insertion or Deletion (Length difference is exactly 1)
    # Ensure s is the longer string for a unified logic (deletion from s)
    if len(s) < len(t):
        s, t = t, s
    
    # Try to find the first point of divergence
    # We use a generator to find the first index where s and t differ
    # We use next() with a default value to avoid StopIteration
    diff_idx = next((i for i in range(len(t)) if s[i] != t[i]), len(t))
    
    # If we remove the character at diff_idx from the longer string s,
    # the remaining string should match t.
    if s[:diff_idx] + s[diff_idx+1:] == t:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()