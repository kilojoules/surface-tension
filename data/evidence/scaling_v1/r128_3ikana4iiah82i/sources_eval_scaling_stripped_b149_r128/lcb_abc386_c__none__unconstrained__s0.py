import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if len(input_data) < 3:
        return
    
    # K is the maximum allowed edit distance
    # S and T are the two strings to compare
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # If strings are already identical, 0 operations are needed
    if s == t:
        print("Yes")
        return
    
    # If the length difference exceeds K, it's impossible to make them identical
    if abs(len(s) - len(t)) > k:
        print("No")
        return

    # Find the first index where the characters of S and T differ.
    # We use a generator expression inside next() to find the first mismatch.
    # If no mismatch is found within the length of the shorter string, 
    # first_diff becomes the length of the shorter string.
    first_diff = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))
    
    # For K=1, we check the three possible edit operations:
    # 1. Replacement: Both strings have the same length, and the suffixes after the mismatch are identical.
    # 2. Deletion from S: S is one character longer than T, and the suffix of S after the mismatch 
    #    matches the suffix of T starting from the mismatch.
    # 3. Insertion into S: T is one character longer than S, and the suffix of S starting from the 
    #    mismatch matches the suffix of T after the mismatch.
    
    # We use a boolean expression to check if any of these conditions are met.
    result = (
        (len(s) == len(t) and s[first_diff + 1:] == t[first_diff + 1:]) or
        (len(s) == len(t) + 1 and s[first_diff + 1:] == t[first_diff:]) or
        (len(t) == len(s) + 1 and s[first_diff:] == t[first_diff + 1:])
    )
    
    # Output Yes if the edit distance is <= 1, otherwise No.
    print("Yes" if result else "No")

# Standard Python entry point
if __name__ == "__main__":
    solve()