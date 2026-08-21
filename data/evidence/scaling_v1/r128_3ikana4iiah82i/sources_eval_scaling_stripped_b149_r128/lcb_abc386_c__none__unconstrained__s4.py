import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The problem is to check if the edit distance between S and T is <= K.
    # Since K=1, we can check this without a full DP table.
    
    # If the length difference is more than K, it's impossible.
    if abs(len(s) - len(t)) > k:
        print("No")
        return

    # If S and T are already identical
    if s == t:
        print("Yes")
        return

    # For K=1, we check the three possible operations:
    # 1. Replacement: Lengths must be equal. 
    #    Check if they differ by exactly one character.
    if len(s) == len(t):
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return
        else:
            print("No")
            return

    # 2. Deletion from S (or Insertion into T): len(s) = len(t) + 1
    # 3. Insertion into S (or Deletion from T): len(s) = len(t) - 1
    # We can generalize this: one string is shorter than the other by 1.
    # We find the first mismatch and check if skipping that character makes them equal.
    
    # Identify which is the longer string
    longer, shorter = (s, t) if len(s) > len(t) else (t, s)
    
    # Find the first index where they differ
    # We use a generator to find the first mismatch index
    first_mismatch = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
    
    # Check if removing the character at first_mismatch from the longer string makes it equal to the shorter one
    # The remaining parts of the strings must be identical.
    # longer[first_mismatch + 1:] must match shorter[first_mismatch:]
    if longer[:first_mismatch] + longer[first_mismatch + 1:] == shorter:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()