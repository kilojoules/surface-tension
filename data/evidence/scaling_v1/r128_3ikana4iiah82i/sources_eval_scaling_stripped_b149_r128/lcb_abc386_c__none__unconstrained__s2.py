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
    # For K=1, we can check this without a full DP table.
    
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
    # 3. Insertion into S (or Deletion from T): len(s) + 1 = len(t)
    # We can handle both by identifying the shorter and longer string.
    short, long = (s, t) if len(s) < len(t) else (t, s)
    
    # For K=1, the only way to transform short to long is by inserting one character.
    # This means short must be a subsequence of long with length len(long) - 1.
    # We find the first index where they differ.
    
    # Find first mismatch
    # Using a generator to find the first index where characters differ
    first_diff = next((i for i in range(len(short)) if short[i] != long[i]), len(short))
    
    # If we insert a character into 'short' at 'first_diff', 
    # the remaining suffixes must be identical.
    # The character to be inserted is long[first_diff].
    # So we check if short[first_diff:] == long[first_diff + 1:]
    if short[first_diff:] == long[first_diff + 1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()