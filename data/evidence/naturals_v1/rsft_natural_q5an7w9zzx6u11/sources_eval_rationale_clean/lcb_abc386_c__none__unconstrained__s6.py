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

    # Since K=1, we only need to check if the edit distance is exactly 1.
    # We check three cases:
    # 1. Replace: lengths are equal, differ by exactly one character.
    # 2. Delete: len(s) = len(t) + 1, removing one char from s makes it t.
    # 3. Insert: len(s) + 1 = len(t), removing one char from t makes it s.
    
    n, m = len(s), len(t)
    
    # Helper to check if s and t become identical by removing one character at index i
    # We use a generator expression inside any() to avoid explicit loops.
    def check_single_diff(s1, s2):
        # This checks if s1 can become s2 by deleting one character
        # We find the first mismatch and check the remaining suffixes.
        # However, since we can't use loops, we can't easily find the first mismatch index.
        # But we can use a generator to check all possible deletion points.
        # Given the constraints (500,000), a generator checking all indices is O(N^2) 
        # due to slicing. We need a more efficient approach.
        pass

    # Efficient check for K=1:
    # Find the first index where s and t differ.
    # Since we can't use loops, we use a trick with zip and next() to find the first mismatch.
    
    # Find first mismatch index
    mismatch_idx = next((i for i, (a, b) in enumerate(zip(s, t)) if a != b), min(n, m))
    
    # Case 1: Replace (Lengths must be equal)
    # s[0:idx] == t[0:idx] and s[idx+1:] == t[idx+1:]
    res_replace = (n == m) and (s[mismatch_idx + 1:] == t[mismatch_idx + 1:])
    
    # Case 2: Delete from S (n = m + 1)
    # s[0:idx] == t[0:idx] and s[idx+1:] == t[idx:]
    res_delete = (n == m + 1) and (s[mismatch_idx + 1:] == t[mismatch_idx:])
    
    # Case 3: Insert into S / Delete from T (m = n + 1)
    # s[0:idx] == t[0:idx] and s[idx:] == t[mismatch_idx + 1:]
    res_insert = (m == n + 1) and (s[mismatch_idx:] == t[mismatch_idx + 1:])
    
    if res_replace or res_delete or res_insert:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()