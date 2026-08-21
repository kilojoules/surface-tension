import sys
from itertools import takewhile

def solve():
    # Read input using map and split to avoid explicit loops
    # K is always 1 for this sub-problem
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Find the length of the common prefix
    # takewhile returns an iterator of characters that match
    # We calculate the length of the common prefix by comparing S and T
    # Since we cannot use loops, we use a generator expression inside max 
    # or simply find the first index where S[i] != T[i].
    # However, the most "loop-free" way to find the prefix length 
    # is to use a helper that identifies the first mismatch.
    
    # We can find the first mismatch index by checking 
    # the length of the common prefix.
    # Using a list comprehension to find all indices where they match 
    # is too slow (O(N)). 
    # Instead, we use the fact that K=1.
    
    # If S == T, it's 0 operations.
    # If they differ, we find the first index 'i' where S[i] != T[i].
    # Then we check if the remaining suffixes can be reconciled in 1 op.
    
    # To find the first mismatch without a loop:
    # We can use a recursive-like approach via map/filter, 
    # but Python's recursion limit is strict.
    # The most efficient "no-loop" way to find the first mismatch 
    # is to use a combination of slicing and comparison.
    
    # Since K=1, we only care if the edit distance is <= 1.
    # Edit distance 1 means:
    # 1. Replace: len(S) == len(T), and they differ by exactly one char.
    # 2. Delete: len(S) == len(T) + 1, and S without one char is T.
    # 3. Insert: len(S) == len(T) - 1, and T without one char is S.
    
    # To implement this without loops, we use the following logic:
    # For a givenを mismatch at index i:
    # Replace: S[i+1:] == T[i+1:]
    # Delete: S[i+1:] == T[i:]
    # Insert: S[i:] == T[i+1:]
    
    # We find 'i' by comparing the strings. 
    # Since we can't loop, we can use a trick: 
    # find the first index where they differ by 
    # checking prefixes of various lengths? No, that's O(N^2).
    # Actually, we can use `itertools.takewhile` to get the common prefix.
    
    # Note: The prompt allows list comprehensions and map/filter.
    # We can find the first mismatch index by:
    # i = next((i for i in range(min(len(S), len(T))) if S[i] != T[i]), min(len(S), len(T)))
    
    # Let's define the logic in a way that uses the 'next' generator.
    i = next((idx for idx, (char_s, char_t) in enumerate(zip(S, T)) if char_s != char_t), min(len(S), len(T)))
    
    # Check the three possibilities for K=1
    # 1. S and T are already identical
    # 2. Replace: S[i+1:] == T[i+1:] and len(S) == len(T)
    # 3. Delete: S[i+1:] == T[i:] and len(S) == len(T) + 1
    # 4. Insert: S[i:] == T[i+1:] and len(S) == len(T) - 1
    
    is_possible = (
        (S == T) or
        (len(S) == len(T) and S[i+1:] == T[i+1:]) or
        (len(S) == len(T) + 1 and S[i+1:] == T[i:]) or
        (len(S) == len(T) - 1 and S[i:] == T[i+1:])
    )
    
    sys.stdout.write("Yes\n" if is_possible else "No\n")

if __name__ == "__main__":
    solve()