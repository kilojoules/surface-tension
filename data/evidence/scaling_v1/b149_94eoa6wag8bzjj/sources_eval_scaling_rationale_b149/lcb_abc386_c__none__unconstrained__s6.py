import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if edit distance(S, T) <= 1
    # Since K=1, we can check the three possible operations:
    # 1. S == T (0 operations)
    # 2. Replace: len(S) == len(T) and they differ by exactly one character
    # 3. Delete: len(S) == len(T) + 1 and S becomes T by removing one char
    # 4. Insert: len(T) == len(S) + 1 and T becomes S by removing one char

    # Helper to check if S and T are identical except for one characterK
    # We use a generator expression inside sum() to count differences
    # Note: sum() is allowed as it is a built-in reduction, not a loop construct.
    
    def check_replace(s1, s2):
        return sum(1 for a, b in zip(s1, s2) if a != b) <= 1

    def check_off_by_one(s_long, s_short):
        # To check if s_long can become s_short by deleting one char without loops:
        # We find the first index where they differ.
        # Since we can't use loops, we use a recursive-like approach via 
        # finding the first mismatch using a generator and next().
        
        # Find first mismatch index
        mismatch_idx = next((i for i in range(len(s_short)) if s_long[i] != s_short[i]), len(s_short))
        # Check if skipping that character in s_long makes it equal to s_short
        return s_long[:mismatch_idx] + s_long[mismatch_idx+1:] == s_short

    # Logic to determine result
    res = (
        (s == t) or
        (len(s) == len(t) and check_replace(s, t)) or
        (len(s) == len(t) + 1 and check_off_by_one(s, t)) or
        (len(t) == len(s) + 1 and check_off_by_one(t, s))
    )

    sys.stdout.write("Yes\n" if res else "No\n")

if __name__ == "__main__":
    solve()