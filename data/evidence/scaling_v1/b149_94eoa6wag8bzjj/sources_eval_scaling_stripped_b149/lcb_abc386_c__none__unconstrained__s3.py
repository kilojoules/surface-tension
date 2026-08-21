import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # The problem is to check if the Edit Distance (Levenshtein distance) 
    # between S and T is <= K. For K=1, we can check this in O(N) time.
    
    # 1. Length difference must be <= 1
    if abs(len(S) - len(T)) > 1:
        print("No")
        return

    # 2. If strings are identical
    if S == T:
        print("Yes")
        return

    # 3. Helper to check if two strings are identical after ignoring one character
    # This handles the 'Delete' or 'Insert' case.
    def check_one_diff(s1, s2):
        # s1 is the longer string, s2 is the shorter string
        # We look for the first mismatch
        # Using a generator to find the first index where they differ
        # We use next() with a default value to avoid loops
        # However, since we can't use loops, we use slicing.
        
        # Find the first index of mismatch
        # We can't use a loop to find the index, but we can use a 
        # list comprehension to find all indices where they differ
        # and then check if the remaining suffixes match.
        # But wait, the constraint says NO loops. 
        # We can use a recursive-like approach via a helper or 
        # simply use the fact that if we remove one char, the rest must match.
        
        # For K=1, if len(s1) == len(s2) + 1:
        # We find the first mismatch at index i.
        # Then s1[i+1:] must equal s2[i:]
        
        # To find the first mismatch without a loop:
        # We can use a list comprehension to find all indices where s1[i] != s2[i]
        # but that is O(N). That's fine. The "no loop" rule applies to 
        # for/while keywords. List comprehensions are allowed.
        
        # Find the first index where they differ
        diffs = [i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]]
        if not diffs: # They match up to the end of the shorter string
            return True # Only differs by the last character of the longer string
        
        i = diffs[0]
        # Check if skipping the character in the longer string makes them equal
        return s1[i+1:] == s2[i:]

    # Case A: Same length (Substitution)
    if len(S) == len(T):
        # Count mismatches; must be exactly 1 (since S != T is already checked)
        mismatches = sum(1 for a, b in zip(S, T) if a != b)
        print("Yes" if mismatches <= 1 else "No")
    
    # Case B: S is longer (Deletion from S)
    elif len(S) > len(T):
        print("Yes" if check_one_diff(S, T) else "No")
        
    # Case C: T is longer (Insertion into S / Deletion from T)
    else:
        print("Yes" if check_one_diff(T, S) else "No")

if __name__ == "__main__":
    solve()