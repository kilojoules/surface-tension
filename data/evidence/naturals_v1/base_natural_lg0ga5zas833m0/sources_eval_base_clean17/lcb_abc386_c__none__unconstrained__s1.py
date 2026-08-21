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

    # If the strings are already identical, 0 operations are needed.
    if s == t:
        print("Yes")
        return

    # Since K=1, we check if the edit distance between S and T is <= 1.
    # Edit distance 1 means:
    # 1. Replacement: lengths are equal, only one character differs.
    # 2. Deletion: len(S) = len(T) + 1, removing one char from S makes it T.
    # 3. Insertion: len(S) = len(T) - 1, adding one char to S makes it T.

    len_s = len(s)
    len_t = len(t)

    # If the difference in length is more than 1, edit distance is > 1.
    if abs(len_s - len_t) > 1:
        print("No")
        return

    if len_s == len_t:
        # Check for replacement: exactly one character difference
        diff_count = 0
        for i in range(len_s):
            if s[i] != t[i]:
                diff_count += 1
                if diff_count > 1:
                    print("No")
                    return
        if diff_count <= 1:
            print("Yes")
        else:
            print("No")

    elif len_s == len_t + 1:
        # Check if deleting one char from S makes it T
        # We find the first mismatch and check the remaining suffix
        i = 0
        while i < len_t and s[i] == t[i]:
            i += 1
        # Try skipping the character at index i in S
        if s[i+1:] == t[i:]:
            print("Yes")
        else:
            print("No")

    elif len_s == len_t - 1:
        # Check if inserting one char into S makes it T
        # This is equivalent to deleting one char from T to make it S
        i = 0
        while i < len_s and s[i] == t[i]:
            i += 1
        # Try skipping the character at index i in T
        if s[i:] == t[i+1:]:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()