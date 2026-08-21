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
    
    # The edit distance between S and T must be <= K.
    # Since K=1, the possible cases are:
    # 0. S == T
    # 1. Length difference is 1:
    #    - S is T with one char inserted (T is S with one char deleted)
    #    - T is S with one char inserted (S is T with one char deleted)
    # 2. Length difference is 0:
    #    - S and T differ by exactly one character (substitution)
    
    # Case 0: Identical
    if s == t:
        print("Yes")
        return

    # Case 1: Length difference is 1 (Insertion or Deletion)
    if abs(n - m) == 1:
        # Ensure 'shorter' is the shorter string and 'longer' is the longer one
        shorter, longer = (s, t) if n < m else (t, s)
        
        # To check if 'longer' can become 'shorter' by deleting one char:
        # We find the first index where they differ.
        # The remaining suffixes must be identical.
        
        # Find first mismatch
        # Using a list comprehension to find the first index of difference
        # We use a trick with next() to avoid loops
        diff_idx = next((i for i in range(len(shorter)) if shorter[i] != longer[i]), len(shorter))
        
        # If we remove the character at diff_idx from 'longer', 
        # the rest should match 'shorter' from diff_idx onwards.
        if shorter[diff_idx:] == longer[diff_idx + 1:]:
            print("Yes")
            return

    # Case 2: Length difference is 0 (Substitution)
    if n == m:
        # Count differences. If exactly 1, it's Yes.
        # Using a generator expression inside sum() to count mismatches.
        diff_count = sum(1 for i in range(n) if s[i] != t[i])
        if diff_count == 1:
            print("Yes")
            return

    # If none of the above conditions are met
    print("No")

if __name__ == "__main__":
    solve()