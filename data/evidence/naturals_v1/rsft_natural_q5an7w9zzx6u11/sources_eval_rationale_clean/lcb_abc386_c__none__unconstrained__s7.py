import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # If strings are already identical, it's 0 operations
    if s == t:
        print("Yes")
        return

    # Since K=1, we only need to check if the edit distance is exactly 1
    # Edit distance 1 means:
    # 1. Substitution: lengths are equal, differ by one character
    # 2. Deletion: len(s) = len(t) + 1, removing one char from s makes it t
    # 3. Insertion: len(s) + 1 = len(t), removing one char from t makes it s

    n, m = len(s), len(t)

    # Case 1: Substitution (Lengths equal)
    # Check if there is exactly one index where s[i] != t[i]
    res_sub = (n == m) and (sum(1 for i in range(n) if s[i] != t[i]) == 1)

    # Case 2: Deletion from S (or Insertion into T)
    # Check if removing one character from S results in T
    # We find the first mismatch and check the remaining suffixes
    # To avoid loops, we can use a helper to find the first mismatch index
    # But since we can't use loops, we use a generator to find the first index
    
    def check_one_diff(s1, s2):
        # This checks if s2 can be formed by removing one char from s1
        # We find the first index where they differ
        # Using a generator to find the first mismatch index
        mismatch_idx = next((i for i in range(len(s2)) if s1[i] != s2[i]), len(s2))
        return s1[mismatch_idx + 1:] == s2[mismatch_idx:]

    # Case 2: len(s) == len(t) + 1
    res_del = (n == m + 1) and check_one_diff(s, t)
    
    # Case 3: len(s) + 1 == len(t)
    res_ins = (n + 1 == m) and check_one_diff(t, s)

    if res_sub or res_del or res_ins:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()