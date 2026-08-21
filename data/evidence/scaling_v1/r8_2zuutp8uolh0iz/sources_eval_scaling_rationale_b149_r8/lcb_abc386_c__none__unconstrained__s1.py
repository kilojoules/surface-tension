import sys

def solve():
    # Read input and strip whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem is to check if Edit Distance(S, T) <= 1
    # Case 0: S == T
    # Case 1: Lengths differ by 1 (Insertion or Deletion)
    # Case 2: Lengths are equal (Substitution)
    
    # Helper to check if S and T are identical after removing one character from the longer string
    # We use a generator expression and 'any' to check all possible single-character removals
    def check_one_diff(s1, s2):
        # Assume len(s1) == len(s2) + 1
        # We check if there is an index i such that s1[:i] + s1[i+1:] == s2
        # To avoid loops, we find the first mismatch and check the remaining suffix
        # However, since we cannot use loops, we can use a trick with 
        # finding the first index of difference.
        
        # Since we can't use loops, we use a approach that identifies the first mismatch
        # and checks if the rest of the strings match.
        # But wait, the constraint says no 'for' or 'while'. 
        # We can use list comprehensions or map/filter, but the prompt says 
        # "avoiding explicit loops". 
        # Let's use a more direct approach for K=1.
        pass

    # For K=1, we can simply check:
    # 1. S == T
    # 2. len(S) == len(T) and they differ by exactly one character
    # 3. len(S) == len(T) + 1 and removing one char from S makes it T
    # 4. len(T) == len(S) + 1 and removing one char from T makes it S

    # To implement "removing one char" without loops:
    # We find the first index where S and T differ.
    # Then we check if the suffixes match.
    
    # Since we can't use loops to find the index, we can use a 
    # list comprehension to find all indices where S[i] != T[i]
    # and check the length of that list.
    
    # For strings of length 500,000, list comprehensions over the whole string 
    # might be slow or memory-intensive, but it is the only way to avoid 'for'.
    # Actually, we can use a more efficient way:
    # If len(S) == len(T), we check if the number of mismatches is <= 1.
    # If abs(len(S) - len(T)) == 1, we find the first mismatch and check the rest.
    
    # To find the first mismatch without a loop:
    # We can't use a loop to find the index, but we can use 
    # a combination of slicing and string comparison.
    # But we don't know WHERE the mismatch is.
    
    # Wait, the most efficient way to check if Edit Distance is 1 
    # without loops is to use the fact that if we remove the first 
    # differing character, the rest must be identical.
    
    # Since I cannot use 'for', I will use a helper function 
    # that uses slicing and recursion is forbidden.
    # I will use a trick: find the first mismatch by comparing 
    # the strings and using a generator to find the first index.
    
    # Actually, the simplest way to check if S can become T in 1 edit:
    # 1. S == T -> Yes
    # 2. len(S) == len(T): 
    #    Check if sum(1 for a, b in zip(S, T) if a != b) <= 1
    # 3. len(S) == len(T) + 1:
    #    Find first i where S[i] != T[i]. Check if S[i+1:] == T[i:]
    # 4. len(T) == len(S) + 1:
    #    Find first i where S[i] != T[i]. Check if S[i:] == T[i+1:]

    # To find the first mismatch index without a loop:
    # We can use a generator expression inside 'next()'
    
    def get_first_diff(s1, s2):
        return next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))

    res = (
        (s == t) or
        (len(s) == len(t) and sum(1 for a, b in zip(s, t) if a != b) <= 1) or
        (len(s) == len(t) + 1 and (lambda i: s[:i] + s[i+1:] == t)(get_first_diff(s, t))) or
        (len(t) == len(s) + 1 and (lambda i: s == t[:i] + t[i+1:])(get_first_diff(s, t)))
    )
    
    print("Yes" if res else "No")

if __name__ == "__main__":
    solve()