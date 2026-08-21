import sys

def solve():
    # Read input using sys.stdin.read().split() to get all tokens
    # The input consists of K, S, and T.
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n = len(s)
    m = len(t)
    
    # The problem asks if the Edit Distance between S and T is <= K.
    # Given K = 1, we can check this without a full DP table.
    
    # Case 0: S and T are already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Length difference is more than 1, impossible for K=1
    if abs(n - m) > 1:
        print("No")
        return

    # Helper to check if two strings are equal after removing one character
    # This handles both Deletion (S -> T) and Insertion (S -> T)
    def check_one_diff(s1, s2):
        # s1 is the longer string, s2 is the shorter string
        # We check if removing one char from s1 makes it s2
        # To do this efficiently without loops, we find the first mismatch
        # and check the remaining suffixes.
        
        # Find the index of the first difference
        # We use a trick with slicing or a comprehension since loops are forbidden
        # But we can use the fact that if we remove one char, the rest must match.
        
        # Since we can't use loops, we can find the first mismatch using a 
        # list comprehension to find indices where s1[i] != s2[i]
        mismatches = [i for i in range(len(s2)) if s1[i] != s2[i]]
        
        if not mismatches:
            # s2 is a prefix of s1
            return True
        
        idx = mismatches[0]
        # If we remove s1[idx], the rest of s1 must match s2 from idx onwards
        return s1[idx+1:] == s2[idx:]

    # Case 2: Substitution (Length is same)
    if n == m:
        # Count differences. If exactly 1, Yes.
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Insertion or Deletion (Length difference is 1)
    # If n > m, it's a deletion from S. If m > n, it's an insertion into S (deletion from T).
    if n > m:
        # Check if removing one char from S gives T
        # To avoid loops, we find the first mismatch
        # We can't use a loop to find the first mismatch, but we can use a 
        # list comprehension to find all mismatches and take the first one.
        mismatches = [i for i in range(m) if s[i] != t[i]]
        idx = mismatches[0] if mismatches else m
        if s[idx+1:] == t[idx:]:
            print("Yes")
        else:
            print("No")
    else: # m > n
        # Check if removing one char from T gives S
        mismatches = [i for i in range(n) if s[i] != t[i]]
        idx = mismatches[0] if mismatches else n
        if s[idx:] == t[idx+1:]:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    solve()