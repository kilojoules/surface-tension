import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if Edit Distance(S, T) <= 1
    # We check 4 cases:
    # 1. S == T (0 operations)
    # 2. Lengths differ by 1: One insertion or one deletion
    # 3. Lengths are equal: One substitution
    
    # Helper to check if S and T are identical after removing one character
    # For K=1, if len(S) == len(T) + 1, we check if T is a subsequence of S
    # However, since we can't use loops, we use a trick with slicing and 
    # finding the first mismatch.
    
    # Since we cannot use loops, we use a recursive-like structure via 
    # a helper function that we call once, but the prompt forbids recursion.
    # We will use a combination of string methods and slicing.
    
    # To check if S can become T by deleting one char from S:
    # Find first index i where S[i] != T[i]. Then S[i+1:] must equal T[i:].
    # We can find the first mismatch using a generator and next().
    
    def check_deletion(long_s, short_t):
        # Find first mismatch
        # We use a generator to find the first index where characters differ
        # next() is allowed as it is a builtin.
        try:
            # Find index of first mismatch
            i = next(idx for idx, (a, b) in enumerate(zip(long_s, short_t)) if a != b)
            return long_s[i+1:] == short_t[i:]
        except StopIteration:
            # One string is a prefix of the other
            return True

    def check_substitution(s1, s2):
        # Count mismatches. Since we can't loop, we use map/sum.
        return sum(map(lambda x: x[0] != x[1], zip(s1, s2))) <= 1

    # Logic for K=1
    res = (
        (s == t) or 
        (len(s) == len(t) and check_substitution(s, t)) or 
        (len(s) == len(t) + 1 and check_deletion(s, t)) or 
        (len(t) == len(s) + 1 and check_deletion(t, s))
    )

    sys.stdout.write("Yes\n" if res else "No\n")

if __name__ == "__main__":
    solve()