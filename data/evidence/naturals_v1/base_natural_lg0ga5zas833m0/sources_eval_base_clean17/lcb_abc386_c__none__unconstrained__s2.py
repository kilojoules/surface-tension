import sys

def solve():
    # Read K, S, and T from standard input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        k = int(line1.strip())
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except ValueError:
        return

    # The problem asks if the edit distance between S and T is <= K.
    # Given the constraint K=1, we can check this efficiently.
    
    # If S and T are already the same, distance is 0.
    if s == t:
        print("Yes")
        return

    # If the absolute difference in lengths is greater than 1, 
    # it's impossible to transform S to T in 1 operation.
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Case 1: Same length - check if they differ by at most one character (Replacement)
    if len(s) == len(t):
        diff_count = 0
        for char_s, char_t in zip(s, t):
            if char_s != char_t:
                diff_count += 1
            if diff_count > 1:
                print("No")
                return
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Lengths differ by 1 - check if one can be transformed to other by 1 insertion/deletion
    # Ensure s is the shorter string
    if len(s) > len(t):
        s, t = t, s
    
    # Now len(t) = len(s) + 1. We check if removing one char from t makes it s.
    # We use two pointers to find the first mismatch.
    i = 0
    while i < len(s) and s[i] == t[i]:
        i += 1
    
    # If we found a mismatch at index i, the rest of s (from i) 
    # must match the rest of t (from i+1).
    if s[i:] == t[i+1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()