import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if edit distance <= 1
    # Case 1: S and T are identical
    # Case 2: Lengths differ by 1 (Insertion/Deletion)
    # Case 3: Lengths are same, differ by 1 char (Replacement)

    # Helper to check if s can become t by deleting one char from s
    # We find the first mismatch and check if the rest of the strings match
    def check_delete(s1, s2):
        # Find first index where they differ
        # Using a generator to find the first mismatch index
        mismatch_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))
        return s1[mismatch_idx + 1:] == s2[mismatch_idx:]

    # Logic for K=1
    # 1. Identical
    # 2. Length difference 0: check if they differ by at most 1 character
    # 3. Length difference 1: check if deleting one char from longer makes it the shorter
    
    result = (
        (s == t) or
        (len(s) == len(t) and sum(1 for a, b in zip(s, t) if a != b) <= 1) or
        (len(s) == len(t) + 1 and check_delete(s, t)) or
        (len(t) == len(s) + 1 and check_delete(t, s))
    )

    print("Yes" if result else "No")

if __name__ == "__main__":
    solve()