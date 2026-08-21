import sys

def solve():
    # Read input and handle potential empty lines/whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # If length difference is more than K, it's impossible
    if abs(n - m) > k:
        print("No")
        return

    # For K=1, we can check if S and T are already identical
    if s == t:
        print("Yes")
        return

    # Since K=1, we only need to check if they differ by exactly one edit.
    # We can use a helper function to check if removing one char from 
    # one string makes it equal to the other, or if changing one char works.
    
    # Case 1: Replace (lengths must be equal)
    if n == m:
        # Count positions where characters differ
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return
        else:
            print("No")
            return

    # Case 2: Insert/Delete (length difference must be exactly 1)
    if abs(n - m) == 1:
        # Ensure s is the shorter string
        shorter, longer = (s, t) if n < m else (t, s)
        
        # Find the first index where they differ
        # We use a generator to find the first mismatch
        first_diff = next((i for i in range(len(shorter)) if shorter[i] != longer[i]), len(shorter))
        
        # If we skip the character at first_diff in the longer string, 
        # the rest must match the shorter string.
        if longer[first_diff + 1:] == shorter[first_diff:]:
            print("Yes")
        else:
            print("No")
        return

    # This part is technically unreachable given the constraints and logic above
    print("No")

if __name__ == "__main__":
    solve()