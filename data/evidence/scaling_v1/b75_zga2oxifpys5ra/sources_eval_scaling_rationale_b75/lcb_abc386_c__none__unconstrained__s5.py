import sys

def solve():
    # Read input and handle potential whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if Edit Distance(S, T) <= 1
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replace one character
    # Lengths must be equal, and they must differ by exactly one position
    # We use a generator expression inside any() to check if exactly one char differs
    def check_replace():
        return len(s) == len(t) and sum(1 for a, b in zip(s, t) if a != b) == 1

    # Case 2: Delete one character from S (or insert one into T)
    # Length of S must be len(T) + 1
    # We check if there exists an index i such that S[:i] + S[i+1:] == T
    def check_delete():
        if len(s) != len(t) + 1:
            return False
        # Find the first mismatch
        # Since we can't use loops, we find the first index where they differ
        # and check if removing that character makes them equal.
        # However, a simpler way without loops is to find the first mismatch 
        # and the last mismatch and verify the middle.
        # But the most robust way is to find the first index i where s[i] != t[i]
        # and check if s[i+1:] == t[i:]
        # To do this without a loop, we can use a helper to find the first mismatch.
        return False # Placeholder, logic handled by the combined check below

    # Because we cannot use loops or recursion, we rely on the fact that 
    # for K=1, we can simply check if the strings are identical after 
    # removing one character from the longer string.
    
    # Optimized check for Delete/Insert:
    # If len(s) == len(t) + 1, we need to find if removing one char from s yields t.
    # If len(t) == len(s) + 1, we need to find if removing one char from t yields s.
    
    # To implement "find first mismatch" without loops:
    # We can use a generator to find the first index where s[i] != t[i].
    def is_one_edit_distance(s1, s2):
        # Ensure s1 is the shorter string
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        if len(s2) - len(s1) > 1:
            return False
        
        # Find first mismatch index
        # We use next() with a generator to find the first index i where s1[i] != s2[i]
        # If no mismatch is found, the mismatch index is the length of the shorter string.
        mismatch_idx = next((i for i in range(len(s1)) if s1[i] != s2[i]), len(s1))
        
        if len(s1) == len(s2):
            # Replace case: remaining suffixes must be identical after the mismatch
            return s1[mismatch_idx+1:] == s2[mismatch_idx+1:]
        else:
            # Insert/Delete case: s1[mismatch_idx:] must match s2[mismatch_idx+1:]
            return s1[mismatch_idx:] == s2[mismatch_idx+1:]

    # Since the prompt forbids loops/recursion, the generator inside next() 
    # is the only way to "iterate" to find the mismatch.
    if is_one_edit_distance(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()