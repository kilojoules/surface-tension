import sys

def solve():
    # Read input
    try:
        input_data = sys.stdin.read().split()
        if not input_data:
            return
        k = int(input_data[0])
        s = input_data[1]
        t = input_data[2]
    except (EOFError, IndexError):
        return

    # The problem asks if the edit distance between S and T is <= K.
    # For this specific sub-problem, K is always 1.
    
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # If K=0 and strings aren't identical, it's impossible.
    # But the constraint says K=1.
    if k == 0:
        print("No")
        return

    n, m = len(s), len(t)

    # Edit distance > 1 if length difference is > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Case 1: Replace one character (lengths must be equal)
    if n == m:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > 1:
                print("No")
                return
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Insert or Delete (length difference is exactly 1)
    # We can treat this as: can we make them equal by removing one char from the longer string?
    if n > m:
        longer, shorter = s, t
    else:
        longer, shorter = t, s

    # Check if removing one character from 'longer' makes it 'shorter'
    # We find the first mismatch
    i = 0
    while i < len(shorter) and longer[i] == shorter[i]:
        i += 1
    
    # After the first mismatch, the rest of the longer string (skipping one char) 
    # must match the rest of the shorter string.
    if longer[i+1:] == shorter[i:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()