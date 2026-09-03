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
    except ValueError:
        return

    # The constraint to avoid explicit loops forces a functional programming style,
    # which can be cleaner for simple transformations but is generally less idiomatic 
    # for complex logic in Python. For this specific problem (K=1), we can 
    # solve it by comparing prefixes and suffixes of the strings.

    n, m = len(s), len(t)

    # If the length difference is more than 1, it's impossible with K=1
    if abs(n - m) > 1:
        print("No")
        return

    # Find the first index where s and t differ
    # We use a generator expression with next() to find the first mismatch
    def get_first_mismatch(s1, s2):
        return next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))

    first = get_first_mismatch(s, t)

    # If they are identical
    if s == t:
        print("Yes")
        return

    # Check if they can be made identical in 1 operation
    # Case 1: Replace (lengths are equal)
    if n == m:
        # If we replace s[first], the remaining suffixes must be identical
        if s[first+1:] == t[first+1:]:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: Delete from S (n = m + 1)
    if n == m + 1:
        # If we delete s[first], the rest of s must match t
        if s[first+1:] == t[first:]:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Insert into S (n = m - 1)
    if n == m - 1:
        # If we insert a char into s at 'first', the rest of s must match t from first+1
        if s[first:] == t[first+1:]:
            print("Yes")
        else:
            print("No")
        return

if __name__ == "__main__":
    solve()