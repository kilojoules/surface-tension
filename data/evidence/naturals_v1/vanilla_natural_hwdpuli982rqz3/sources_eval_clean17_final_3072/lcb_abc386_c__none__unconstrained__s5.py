import sys

def solve():
    # Read input using sys.stdin.read().split() to avoid explicit loops
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance between S and T must be <= K.
    # Since K=1, we check if S == T (0 operations) or if they differ by 1 operation.
    
    # Case 0: Identical
    if s == t:
        print("Yes")
        return

    # Case 1: Length difference > 1 means edit distance > 1
    if abs(n - m) > 1:
        print("No")
        return

    # Case 2: Lengths are equal (Substitution)
    if n == m:
        # Count positions where characters differ
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Lengths differ by 1 (Insertion or Deletion)
    # Ensure 's' is the shorter string for simplicity
    if n > m:
        s, t = t, s
        n, m = m, n
    
    # S is shorter than T by 1. We check if we can make S equal to T by inserting one char into S.
    # This is equivalent to checking if S is a subsequence of T with only one character removed from T.
    # We find the first index where they differ.
    
    # Find the first mismatch
    # Using a list comprehension to find the first index of difference
    # Since we can't use loops, we use a trick with slicing or a specific logic.
    # However, we can find the first mismatch by comparing the strings.
    
    # To avoid loops, we can use a technique to find the first difference:
    # We find the length of the common prefix and common suffix.
    
    # Common prefix length
    # We can't use a loop, but we can use a binary search or simply 
    # check if S is equal to T[0:n] or T[1:m] or T[0:n-1]... 
    # Actually, the most straightforward way to check if edit distance is 1 when |n-m|=1:
    # S must be equal to T after removing one character from T.
    # T[0:i] + T[i+1:m] == S for some i.
    
    # To do this without a loop, we find the first index 'i' where S[i] != T[i].
    # Then S[i:] must be equal to T[i+1:].
    
    # Finding the first mismatch index without a loop:
    # We can use a approach with map/filter or just a clever slice.
    # Since we need to find the FIRST index, we can't easily avoid a loop 
    # unless we use recursion (forbidden) or specific functions.
    # Wait, the constraint says "Write a complete Python program". 
    # It doesn't forbid 'for' or 'while' loops. It forbids "recursion" in some contexts 
    # but usually, loops are the standard way to solve this.
    
    # Let's use a while loop to find the first mismatch.
    i = 0
    while i < n and s[i] == t[i]:
        i += 1
    
    # If the remaining part of S matches the remaining part of T (shifted by 1)
    if s[i:] == t[i+1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()