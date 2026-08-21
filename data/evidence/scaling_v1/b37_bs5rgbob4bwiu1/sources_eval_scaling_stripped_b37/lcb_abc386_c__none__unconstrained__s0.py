import sys

def solve():
    # Read input and strip whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    n, m = len(S), len(T)
    
    # The problem asks if the edit distance between S and T is <= K.
    # Given K=1, we can check this efficiently without a full DP table.
    
    # 1. If lengths differ by more than 1, edit distance is definitely > 1.
    if abs(n - m) > 1:
        print("No")
        return

    # 2. If strings are identical, distance is 0.
    if S == T:
        print("Yes")
        return

    # 3. Case: Replace (lengths are equal)
    if n == m:
        # Count positions where characters differ.
        # If exactly 1 position differs, distance is 1.
        diffs = [i for i in range(n) if S[i] != T[i]]
        if len(diffs) == 1:
            print("Yes")
        else:
            print("No")
        return

    # 4. Case: Insert/Delete (lengths differ by exactly 1)
    # Ensure S is the shorter string for a generalized "insertion" check.
    if n > m:
        S, T = T, S
        n, m = m, n
    
    # Now len(T) = len(S) + 1. We check if T can be formed by inserting 1 char into S.
    # This is equivalent to checking if S is a subsequence of T with only one character difference.
    # We find the first index where S and T differ.
    i = 0
    while i < n and S[i] == T[i]:
        i += 1
    
    # If we skip the differing character in T, the rest of S must match the rest of T.
    # S[i:] == T[i+1:]
    if S[i:] == T[i+1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()