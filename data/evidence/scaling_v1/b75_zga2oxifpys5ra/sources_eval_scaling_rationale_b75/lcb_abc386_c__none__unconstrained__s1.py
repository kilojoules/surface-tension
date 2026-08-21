import sys

def solve():
    # Read input and handle potential whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if edit distance <= 1
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # Since K=1, we check if one operation can make them equal.
    # We can use generator expressions with any() to avoid explicit loops.
    
    # Case 1: Replacement (Lengths must be equal)
    # Check if they differ by exactly one character
    res_replace = (
        len(s) == len(t) and 
        sum(1 for a, b in zip(s, t) if a != b) == 1
    )

    # Case 2: Deletion from S (or Insertion into T)
    # S is longer than T by 1. Try removing one char from S.
    # We find the first mismatch and check if the rest matches.
    # To avoid loops, we can use a trick: find the first index where they differ.
    # However, since we can't use loops, we can't easily find the index.
    # But we can use a generator to check all possible single deletions.
    # Given the constraint of 500,000 characters, a generator checking all 
    # slices would be O(N^2). We need an O(N) approach.
    
    # Optimized O(N) check for K=1:
    # Two strings are distance 1 if:
    # 1. Lengths differ by 1: One is a subsequence of the other.
    # 2. Lengths are same: They differ by one character.
    
    def check_one_diff(s1, s2):
        # Checks if s2 can be formed by deleting one char from s1
        # We find the first mismatch and compare the remaining suffixes.
        # Since we can't use loops, we use a helper logic.
        # But wait, the constraint says no loops/recursion. 
        # We can use a trick with zip and next() to find the first mismatch.
        
        # Find index of first mismatch
        mismatch_idx = next((i for i, (a, b) in enumerate(zip(s1, s2)) if a != b), None)
        
        if mismatch_idx is None:
            # One string is a prefix of the other
            return abs(len(s1) - len(s2)) == 1
        
        # If mismatch found at i, check if s1[i+1:] == s2[i:] (deletion from s1)
        # or s1[i:] == s2[i+1:] (deletion from s2)
        return (s1[mismatch_idx + 1:] == s2[mismatch_idx:] or 
                s1[mismatch_idx:] == s2[mismatch_s2_idx := mismatch_idx + 1:]) if 'mismatch_s2_idx' in locals() else False

    # Let's redefine the logic without a helper function to keep it flat and loop-free.
    # For K=1, the strings are distance 1 if:
    # 1. len(s) == len(t) and they differ by 1 char.
    # 2. len(s) == len(t) + 1 and s becomes t after 1 deletion.
    # 3. len(t) == len(s) + 1 and t becomes s after 1 deletion.

    # To implement the "deletion" check without loops or recursion:
    # We find the first index i where s[i] != t[i].
    # Then we check if s[i+1:] == t[i:] or s[i:] == t[i+1:].
    
    # Using a list comprehension to find the first mismatch index:
    # We use next() on a generator to find the first index of difference.
    diff_idx = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))
    
    # Check if removing one char makes them equal
    # Case: Replace
    cond_replace = (len(s) == len(t) and sum(1 for i in range(len(s)) if s[i] != t[i]) == 1)
    
    # Case: Delete from S / Insert into T
    cond_del_s = (len(s) == len(t) + 1 and s[diff_idx + 1:] == t[diff_idx:])
    
    # Case: Delete from T / Insert into S
    cond_del_t = (len(t) == len(s) + 1 and s[diff_idx:] == t[diff_idx + 1:])
    
    # Case: One is prefix of other and length difference is 1
    cond_prefix = (abs(len(s) - len(t)) == 1 and (s == t + s[-1] if len(s) > len(t) else t == s + t[-1]))
    # Actually, the diff_idx logic handles the prefix case if we are careful.
    # If diff_idx == len(shorter), then one is a prefix of the other.
    # In that case, we just need to check if length difference is 1.
    cond_edge = (abs(len(s) - len(t)) == 1 and (s[:len(t)] == t or t[:len(s)] == s))

    if cond_replace or cond_del_s or cond_del_t or cond_edge:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()