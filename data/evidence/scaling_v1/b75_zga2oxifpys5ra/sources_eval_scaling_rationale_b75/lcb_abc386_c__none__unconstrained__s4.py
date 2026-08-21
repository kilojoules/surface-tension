import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return

    # Since K=1, we only need to check if one operation suffices.
    # We check three scenarios:
    # 1. Substitution: Lengths are equal, and they differ by exactly one character.
    # 2. Deletion from S: len(S) = len(T) + 1, and T is a subsequence of S.
    # 3. Insertion into S: len(T) = len(S) + 1, and S is a subsequence of T.

    # Helper to check if removing one character from 'longer' makes it 'shorter'
    # We find the first mismatch and check if the rest of the strings match.
    def check_one_diff(s1, s2):
        # Find the first index where characters differ
        # We use a generator expression with next() to find the first mismatch index
        mismatch_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        if mismatch_idx is None:
            # One string is a prefix of the other
            return abs(len(s1) - len(s2)) <= 1
        
        # If they differ at mismatch_idx:
        # For substitution: s1[idx+1:] == s2[idx+1:] (and lengths must be equal)
        # For deletion/insertion: s1[idx+1:] == s2[idx:] or s1[idx:] == s2[idx+1:]
        return (
            (len(s1) == len(s2) and s1[mismatch_idx+1:] == s2[mismatch_idx+1:]) or
            (len(s1) > len(s2) and s1[mismatch_idx+1:] == s2[mismatch_idx:]) or
            (len(s2) > len(s1) and s1[mismatch_idx:] == s2[mismatch_idx+1:])
        )

    # We use a conditional expression to print Yes/No based on the check
    print("Yes" if check_one_diff(s, t) else "No")

if __name__ == "__main__":
    solve()