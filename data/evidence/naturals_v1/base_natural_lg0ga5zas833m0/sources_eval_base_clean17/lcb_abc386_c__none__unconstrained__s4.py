import sys

def solve():
    # Read K, S, and T from standard input
    try:
        input_data = sys.stdin.read().split()
        if not input_data:
            return
        k = int(input_data[0])
        s = input_data[1]
        t = input_data[2]
    except EOFError:
        return

    # The problem states K=1 for this sub-problem.
    # We need to check if the edit distance (Levenshtein distance) between S and T is <= K.
    
    # Case 0: S and T are already identical
    if s == t:
        print("Yes")
        return

    # If the difference in lengths is more than K, it's impossible
    if abs(len(s) - len(t)) > k:
        print("No")
        return

    # Since K=1, we only need to check if we can transform S to T in exactly 1 operation.
    # Possible operations: 1 replacement, 1 insertion, or 1 deletion.
    
    n, m = len(s), len(t)
    
    # Case 1: Replacement (lengths must be equal)
    if n == m:
        diff_count = 0
        for i in range(n):
            if s[i] != t[i]:
                diff_count += 1
            if diff_count > k:
                break
        if diff_count <= k:
            print("Yes")
            return

    # Case 2: Deletion from S (n = m + 1)
    if n == m + 1:
        # Try to find if removing one char from S makes it T
        # We find the first mismatch
        i = 0
        while i < m and s[i] == t[i]:
            i += 1
        # Check if the rest of S (skipping one char) matches the rest of T
        if s[i+1:] == t[i:]:
            print("Yes")
            return

    # Case 3: Insertion into S (n = m - 1)
    if n == m - 1:
        # This is symmetric to deletion from T
        i = 0
        while i < n and s[i] == t[i]:
            i += 1
        # Check if the rest of S matches the rest of T (skipping one char in T)
        if s[i:] == t[i+1:]:
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()