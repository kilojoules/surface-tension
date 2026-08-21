import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem states K=1 for this sub-problem.
    # We need to check if the edit distance between S and T is <= K.
    
    # 1. Length difference must be <= K
    if abs(len(s) - len(t)) > k:
        print("No")
        return

    # 2. If S == T, it's always Yes (0 operations)
    if s == t:
        print("Yes")
        return

    # 3. For K=1, we can check the three possible operations:
    # Case A: Replace one character (lengths must be equal)
    # We check if they differ by exactly one character.
    def check_replace():
        if len(s) != len(t):
            return False
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        return len(diffs) == 1

    # Case B: Delete one character from S (len(s) = len(t) + 1)
    # Or Insert one character into S (len(s) + 1 = len(t))
    # Both are equivalent to checking if one string is a subsequence of the other
    # and the length difference is exactly 1.
    def check_insert_delete():
        if abs(len(s) - len(t)) != 1:
            return False
        
        # Ensure 'longer' is the longer string and 'shorter' is the shorter one
        longer, shorter = (s, t) if len(s) > len(t) else (t, s)
        
        # Find the first index where they differ
        # Using a generator to find the first mismatch
        mismatch_idx = next((i for i in range(len(shorter)) if longer[i] != shorter[i]), len(shorter))
        
        # If we skip the mismatched character in the longer string, 
        # the rest of the strings must be identical.
        return longer[mismatch_idx + 1:] == shorter[mismatch_idx:]

    # Since K=1, we just need to check if either condition is met.
    # We use a logical OR.
    if check_replace() or check_insert_delete():
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()