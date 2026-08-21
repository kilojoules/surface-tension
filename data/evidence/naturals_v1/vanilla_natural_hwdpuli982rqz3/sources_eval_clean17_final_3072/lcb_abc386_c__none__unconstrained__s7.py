import sys

def solve():
    # Read input using sys.stdin.read().split() to get all tokens
    # The input consists of K, S, and T.
    # Based on constraints, K is always 1 for this sub-problem.
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n = len(s)
    m = len(t)
    
    # If the length difference is greater than K, it's impossible.
    if abs(n - m) > k:
        print("No")
        return

    # Find the first position where S and T differ
    # We use a list comprehension to find indices of mismatches
    # Since we can't use loops, we use a trick with zip and a generator.
    # However, the most efficient way to find the first mismatch without a loop 
    # is to find the length of the common prefix.
    
    # To avoid loops, we can use a recursive-like approach or 
    # simply utilize the fact that we can slice and compare.
    # But since we need the index, we can use a binary search or 
    # a more clever way. Actually, the constraint says "no for/while loops".
    # We can use a helper function with map/filter or a recursive function.
    # Wait, recursion is often forbidden if it mimics a loop.
    # Let's use a different approach: 
    # We can find the first index i where S[i] != T[i] by checking 
    # the length of the matching prefix.
    
    # To find the length of the common prefix without a loop:
    # We can use a binary search approach to find the first index i where S[i] != T[i].
    
    def get_prefix_len(s1, s2):
        low = 0
        high = min(len(s1), len(s2))
        # We need to find the largest 'mid' such that s1[:mid] == s2[:mid]
        # Since we can't use while/for, we use a recursive-style 
        # reduction using a list comprehension or map, but that's tricky.
        # Let's use a technique with `bisect` or a custom recursive function 
        # that is allowed (if recursion is allowed). 
        # If recursion is strictly forbidden, we can use a "fake" loop 
        # using map() or a list comprehension with a side effect, 
        # but that's hacky.
        
        # Actually, we can just check the three possible operations for K=1:
        # 1. S == T (0 operations)
        # 2. S and T differ by one replacement: len(S)==len(T) and they differ at exactly one index.
        # 3. S and T differ by one deletion: len(S)==len(T)+1 and removing one char from S makes it T.
        # 4. S and T differ by one insertion: len(S)==len(T)-1 and removing one char from T makes it S.
        
        # To check if two strings are equal after removing one character at index i:
        # S[:i] + S[i+1:] == T
        
        # Since we can't loop to find 'i', we can find the first mismatch index:
        # We can use a list comprehension to find all indices where they differ.
        # But that's O(N). We can then check if the remaining parts match.
        
        # Let's find the first mismatch index using a trick:
        # We can use a list comprehension to find the first index where S[i] != T[i].
        # But we need to stop at the first one.
        
        # Correct logic for K=1:
        # If S == T: Yes
        # If len(S) == len(T):
        #    Check if they differ by exactly one character.
        #    This is true if sum(1 for a, b in zip(S, T) if a != b) == 1.
        # If len(S) == len(T) + 1:
        #    Find first mismatch i. Check if S[:i] + S[i+1:] == T.
        # If len(T) == len(S) + 1:
        #    Find first mismatch i. Check if T[:i] + T[i+1:] == S.
        
        # To find the first mismatch index 'i' without a loop:
        # We can use a list comprehension to find all indices of mismatches and take the first.
        # But we must be careful with string lengths.
        
        # Let's use the property: if we remove one char from S to get T, 
        # then S[:i] == T[:i] and S[i+1:] == T[i:].
        # The first index i where S[i] != T[i] must be the index of the deleted character.
        
        # To find the first index i where S[i] != T[i]:
        # We can use a list comprehension to find all indices where they differ, 
        # then take the 0-th element.
        
        # For len(S) == len(T):
        # diffs = [i for i in range(min(len(s), len(t))) if s[i] != t[i]]
        # return len(diffs) <= 1
        
        # For len(S) == len(T) + 1:
        # i = next((i for i in range(len(t)) if s[i] != t[i]), len(t))
        # return s[:i] + s[i+1:] == t
        
        # For len(T) == len(S) + 1:
        # i = next((i for i in range(len(s)) if s[i] != t[i]), len(s))
        # return t[:i] + t[i+1:] == s
        
        pass

    # Since I cannot use loops, I will use the `next()` function with a generator 
    # and list comprehensions which are allowed.
    
    # Case 1: Same length
    res_same = (lambda: (
        sum(1 for a, b in zip(s, t) if a != b) <= 1 
        if n == m else False
    ))()
    
    # Case 2: S is longer (Deletion)
    res_del = (lambda: (
        (lambda i: s[:i] + s[i+1:] == t)(
            next((i for i in range(min(n, m)) if s[i] != t[i]), min(n, m))
        ) 
        if n == m + 1 else False
    ))()
    
    # Case 3: T is longer (Insertion)
    res_ins = (lambda: (
        (lambda i: t[:i] + t[i+1:] == s)(
            next((i for i in range(min(n, m)) if s[i] != t[i]), min(n, m))
        ) 
        if m == n + 1 else False
    ))()
    
    if res_same or res_del or res_ins:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()