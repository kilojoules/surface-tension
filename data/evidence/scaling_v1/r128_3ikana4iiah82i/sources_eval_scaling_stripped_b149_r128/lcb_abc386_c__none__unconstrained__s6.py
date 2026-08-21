import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance between S and T is <= K.
    # Since K is small (specifically K=1 for this sub-problem), 
    # we can check the three possible operations.

    # Case 0: S is already identical to T
    if s == t:
        print("Yes")
        return

    # Case 1: Replace one character
    # S and T must have the same length and differ by exactly one character.
    if len(s) == len(t):
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return

    # Case 2: Delete one character from S (or insert one into T)
    # S must be longer than T by 1.
    if len(s) == len(t) + 1:
        # Find the first index where they differ
        # We can use a generator to find the first mismatch
        # But since we can't use loops, we use a trick with slicing.
        # We check if removing one char from S makes it T.
        # However, we don't know which char to remove.
        # We can find the first mismatch index 'i' and check if S[:i] + S[i+1:] == T.
        
        # To find the first mismatch without a loop:
        # We can use a list comprehension to find all indices where they differ
        # and check if the remaining parts match.
        # But the simplest way for K=1 is to check if T is a subsequence of S.
        # Since len(S) = len(T) + 1, if T is a subsequence, the edit distance is 1.
        
        # Checking subsequence for K=1:
        # We find the first index i where S[i] != T[i].
        # Then we check if S[i+1:] == T[i:].
        # To do this without loops, we can use a helper logic.
        pass

    # Let's redefine the logic to be more "functional" to avoid loops.
    # For K=1, the distance is 1 if:
    # 1. len(S) == len(T) and they differ by 1 char.
    # 2. len(S) == len(T) + 1 and T is a subsequence of S.
    # 3. len(T) == len(S) + 1 and S is a subsequence of T.

    # Helper to check if short is a subsequence of long when len(long) == len(short) + 1
    def check_sub(short, long):
        # Find first mismatch
        # We use a list comprehension to find the first index where they differ.
        # Since we can't use a loop, we can use a trick with 'next' and a generator.
        try:
            i = next(i for i in range(len(short)) if short[i] != long[i])
            return long[:i] + long[i+1:] == short
        except StopIteration:
            # They matched up to the end of 'short', so the extra char is at the end of 'long'
            return True

    # We can't use 'for' or 'while', but we can use 'next' with a generator expression.
    # The constraints allow list comprehensions and generator expressions.
    
    # Re-evaluating the logic inside a single print statement using a conditional expression.
    # We need to handle the subsequence check carefully.
    
    # For the subsequence check:
    # If len(S) == len(T) + 1, we check if S[i+1:] == T[i:] at the first mismatch.
    # We can use a helper function defined inside solve() using def.
    
    # Let's use the logic:
    # Result is "Yes" if:
    # (len(S) == len(T) and sum(1 for a, b in zip(S, T) if a != b) == 1) OR
    # (len(S) == len(T) + 1 and check_sub(T, S)) OR
    # (len(T) == len(S) + 1 and check_sub(S, T))
    
    # Since I cannot use loops, I will use the 'next' function to find the mismatch.
    # I'll wrap the logic in a way that it executes.
    
    # To avoid the 'def' inside 'print', I'll just use the logic directly.
    # For the subsequence check: 
    # if len(S) == len(T) + 1:
    #    mismatch_idx = next((i for i in range(len(T)) if S[i] != T[i]), len(T))
    #    is_sub = (S[:mismatch_idx] + S[mismatch_idx+1:] == T)
    
    # Putting it all together:
    
    res = (
        (len(s) == len(t) and sum(1 for a, b in zip(s, t) if a != b) == 1) or
        (len(s) == len(t) + 1 and (lambda s, t: (lambda i: s[:i] + s[i+1:] == t)(next((i for i in range(len(t)) if s[i] != t[i]), len(t))))(s, t)) or
        (len(t) == len(s) + 1 and (lambda s, t: (lambda i: t[:i] + t[i+1:] == s)(next((i for i in range(len(s)) if t[i] != s[i]), len(s))))(s, t))
    )
    
    print("Yes" if res else "No")

if __name__ == "__main__":
    solve()