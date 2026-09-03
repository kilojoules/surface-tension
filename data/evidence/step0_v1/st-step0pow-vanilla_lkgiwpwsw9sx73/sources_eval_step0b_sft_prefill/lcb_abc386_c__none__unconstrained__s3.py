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
    # Since K=1, we check if S == T or if they can be made equal in one operation.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Difference in length is more than 1
    if abs(n - m) > 1:
        print("No")
        return

    # Case 2: Same length (Substitution)
    if n == m:
        # Count positions where characters differ
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: Length difference is 1 (Insertion or Deletion)
    # Ensure 's' is the shorter string for simplicity
    short, long = (s, t) if n < m else (t, s)
    
    # Find the first index where they differ
    # We can use a trick with slicing or a loop. 
    # Since we can't use loops, we find the first mismatch using a list comprehension
    # and then check if the rest of the strings match after skipping one char in 'long'.
    
    # Find the first index of mismatch
    # We use a generator inside next() to find the first index where short[i] != long[i]
    try:
        first_diff = next(i for i in range(len(short)) if short[i] != long[i])
    except StopIteration:
        # One string is a prefix of the other
        print("Yes")
        return

    # If we remove the character at first_diff from 'long', does it become 'short'?
    # long[:first_diff] + long[first_diff+1:] == short
    if long[:first_diff] + long[first_diff+1:] == short:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()