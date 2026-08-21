import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem states K=1 for this specific sub-problem.
    # The distance required is the Levenshtein distance.
    # For K=1, we check if the edit distance between S and T is <= 1.
    
    n = len(s)
    m = len(t)
    
    # If length difference is more than 1, edit distance must be > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Case 1: S and T are identical
    if s == t:
        print("Yes")
        return

    # Case 2: One character replacement (lengths must be equal)
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

    # Case 3: Insertion or Deletion (length difference is exactly 1)
    # Ensure s is the shorter string for simplicity
    if n > m:
        shorter, longer = t, s
    else:
        shorter, longer = s, t
    
    # Check if shorter is a subsequence of longer by removing exactly one char
    # We use two pointers to find the first mismatch
    i = 0
    while i < len(shorter) and shorter[i] == longer[i]:
        i += 1
    
    # After the first mismatch, the rest of shorter must match the rest of longer shifted by 1
    # shorter[i:] == longer[i+1:]
    if shorter[i:] == longer[i+1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()