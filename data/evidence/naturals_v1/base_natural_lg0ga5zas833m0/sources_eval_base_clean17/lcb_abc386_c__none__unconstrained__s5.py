import sys

def solve():
    # Read input
    try:
        line1 = sys.stdin.readline()
        if not line1:
            return
        k = int(line1.strip())
        s = sys.stdin.readline().strip()
        t = sys.stdin.readline().strip()
    except EOFError:
        return

    # The problem asks if the edit distance between S and T is <= K.
    # Specifically for this sub-problem, K = 1.
    
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # Case 1: K=0 is handled by s == t. If K=1, we check if edit distance is exactly 1.
    # If K is larger, this specific logic only works for K=1.
    # The constraints say K=1 for this sub-problem.
    
    n, m = len(s), len(t)
    
    # If the length difference is more than 1, edit distance must be > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Check if they can be made identical with 1 operation
    # Operation 1: Replace (lengths must be equal)
    if n == m:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > 1:
                break
        if diff_count <= 1:
            print("Yes")
            return

    # Operation 2: Delete from S / Insert into S
    # This covers cases where len(S) = len(T) + 1 or len(S) + 1 = len(T)
    if abs(n - m) == 1:
        # Ensure 'shorter' is the string with length min(n, m)
        shorter = s if n < m else t
        longer = t if n < m else s
        
        # Try to find the single character difference
        i = 0
        while i < len(shorter) and shorter[i] == longer[i]:
            i += 1
        
        # If we skip one character in 'longer', the rest must match 'shorter'
        if shorter[i:] == longer[i+1:]:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()