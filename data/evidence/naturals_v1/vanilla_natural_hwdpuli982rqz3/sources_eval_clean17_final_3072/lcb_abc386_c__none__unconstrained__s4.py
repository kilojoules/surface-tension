import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n = len(s)
    m = len(t)
    
    # The edit distance between S and T must be <= K.
    # Since K=1, we check if S == T or if they can be made equal in 1 operation.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replace one character
    # Lengths must be equal, and they must differ by exactly one character.
    if n == m:
        # Count differences between s and t
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        if diffs <= k:
            print("Yes")
            return

    # Case 2: Delete one character from S (or insert one into T)
    # Length of S must be length of T + 1
    if n == m + 1:
        # Check if removing one char from S makes it T
        # We find the first mismatch and check the rest
        # Using a trick: find the first index where they differ
        # The remaining parts of S (from i+1) and T (from i) must be identical.
        
        # Find first mismatch
        # We can't use a loop, but we can use slicing and comparison
        # To avoid loops, we can find the length of the common prefix
        # However, since we can't use loops, we can use a different approach:
        # If we remove S[i], then S[0:i] == T[0:i] and S[i+1:] == T[i:]
        # We can find the first index i where S[i] != T[i].
        # But since we can't loop, we can use a list comprehension to find the first mismatch.
        
        # Find the index of the first difference
        # We use a generator inside next() to find the first index i where s[i] != t[i]
        # If no difference is found, the difference is at the last character of S.
        try:
            first_diff = next(i for i in range(m) if s[i] != t[i])
        except StopIteration:
            first_diff = m
            
        if s[first_diff + 1:] == t[first_diff:]:
            print("Yes")
            return

    # Case 3: Insert one character into S (or delete one from T)
    # Length of T must be length of S + 1
    if m == n + 1:
        # Similar to Case 2, but roles of S and T are swapped
        try:
            first_diff = next(i for i in range(n) if s[i] != t[i])
        except StopIteration:
            first_diff = n
            
        if s[first_diff:] == t[first_diff + 1:]:
            print("Yes")
            return

    # If none of the above conditions are met
    print("No")

if __name__ == "__main__":
    solve()