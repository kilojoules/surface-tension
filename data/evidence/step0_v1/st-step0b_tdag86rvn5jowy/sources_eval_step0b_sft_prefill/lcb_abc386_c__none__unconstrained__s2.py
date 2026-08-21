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
    # Since K=1, we check if S == T or if they can be made equal in 1 operation.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replace (lengths must be equal)
    # Two strings are 1 replacement apart if they differ by exactly one character.
    if n == m:
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return
        else:
            print("No")
            return

    # Case 2: Delete/Insert (length difference must be exactly 1)
    if abs(n - m) == 1:
        # Identify which is shorter and which is longer
        shorter = s if n < m else t
        longer = t if n < m else s
        
        # Find the first index where they differ
        # We can use a trick: if we remove one char from 'longer', does it become 'shorter'?
        # To do this in O(N), we find the first mismatch.
        
        # Find the index of the first difference
        # Using a list comprehension to find the first index where they differ
        # Since we can't use loops, we use a combination of slicing and indexing.
        
        # We need to find the first i such that shorter[i] != longer[i]
        # We can use a helper to find the length of the common prefix.
        
        # To avoid loops, we can use a technique to find the first mismatch:
        # However, the constraint allows us to check if removing one char works.
        # Since we can't loop, we can find the first mismatch using a method that 
        # doesn't explicitly loop through the whole string in Python code.
        
        # Actually, we can find the first mismatch by comparing slices or using 
        # a method that finds the first difference.
        # But wait, we can just check if the strings are equal after skipping one char.
        
        # Let's find the first index of difference:
        # We can't use a loop, but we can use a list comprehension to find all indices 
        # where they differ, but that's O(N). 
        # Let's use the property: if we remove one char from 'longer', 
        # the remaining parts must match 'shorter'.
        
        # To find the mismatch index without a loop:
        # We can use a binary search or just accept that we need to find the first i.
        # But we can use a trick: 
        # If we remove the character at index i, then longer[0:i] == shorter[0:i] and longer[i+1:] == shorter[i:].
        # The first index i where they differ is the only candidate for removal.
        
        # To find the first index i where shorter[i] != longer[i]:
        # We can use a list comprehension to find all indices where they differ, 
        # but that's not allowed? No, list comprehensions are allowed.
        
        # Let's find the first index of difference:
        # We use a dummy value if no difference is found.
        diff_indices = [i for i in range(min(n, m)) if shorter[i] != longer[i]]
        
        if not diff_indices:
            # They are identical up to the length of the shorter string.
            # Removing the last char of the longer string makes them equal.
            print("Yes")
        else:
            # The first difference is at diff_indices[0].
            # We must remove the character at this index from the longer string.
            i = diff_indices[0]
            if longer[:i] + longer[i+1:] == shorter:
                print("Yes")
            else:
                print("No")
        return

    # Case 3: Length difference > 1
    print("No")

if __name__ == "__main__":
    solve()