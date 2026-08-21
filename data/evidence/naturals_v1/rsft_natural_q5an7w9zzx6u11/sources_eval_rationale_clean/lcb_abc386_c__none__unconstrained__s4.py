import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance is <= 1 if:
    # 1. S and T are identical (distance 0)
    # 2. S and T differ by one character replacement (distance 1)
    # 3. S can be transformed to T by deleting one char from S (distance 1)
    # 4. S can be transformed to T by inserting one char into S (distance 1)
    
    # Case: Identical
    if s == t:
        print("Yes")
        return

    # Case: Replacement (lengths must be equal)
    # We count how many characters differ at the same index.
    if n == m:
        diff_count = sum(1 for i in range(n) if s[i] != t[i])
        if diff_count <= k:
            print("Yes")
            return

    # Case: Deletion from S / Insertion into S
    # If length difference is exactly 1, we check if removing one char makes them equal.
    # We find the first index where they differ and check the remaining suffixes.
    if abs(n - m) == 1:
        # Ensure s is the longer string for a unified deletion check
        s_long, t_short = (s, t) if n > m else (t, s)
        
        # Find the first index of mismatch
        # We use a generator to find the first index where characters differ
        # Since we can't use a loop, we find the mismatch index by comparing 
        # the strings and finding the first index where they differ.
        # However, a simpler way without loops is to check if t_short is a 
        # subsequence of s_long with only one character removed.
        
        # To avoid loops, we can't easily find the 'first' mismatch index 
        # without a loop or recursion. But we can use a trick:
        # If we remove one character from s_long, does it become t_short?
        # This is true if there exists an index i such that s_long[:i] + s_long[i+1:] == t_short.
        # But we can't loop through i. 
        # Instead, we find the first mismatch using a generator and next().
        try:
            first_mismatch = next(i for i in range(min(n, m)) if s[i] != t[i])
            
            # Check if skipping the mismatch in the longer string aligns the rest
            # If s is longer, skip s[first_mismatch]
            # If t is longer, skip t[first_mismatch]
            res = (s[first_mismatch+1:] == t[first_mismatch:]) if n > m else (s[first_mismatch:] == t[first_mismatch+1:])
            
            # Special case: the mismatch is at the very end
            # The logic above covers it, but we must handle the case where 
            # the mismatch is the last character of the longer string.
            # Actually, the slice logic handles it.
            
            if res:
                print("Yes")
                return
        except StopIteration:
            # One string is a prefix of the other and length difference is 1
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    solve()