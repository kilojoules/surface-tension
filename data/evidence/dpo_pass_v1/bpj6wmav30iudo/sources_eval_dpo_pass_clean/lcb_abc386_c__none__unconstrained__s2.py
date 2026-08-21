import sys
from itertools import takewhile

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # Find the length of the common prefix
    # takewhile returns an iterator of characters that match
    # We use a generator expression to pair characters and find where they stop matching
    common_prefix_len = len(list(takewhile(lambda x: x[0] == x[1], zip(S, T))))

    # Suffixes after the common prefix
    s_suffix = S[common_prefix_len:]
    t_suffix = T[common_prefix_len:]

    # For K=1, we check the three possible edit operations:
    # 1. Replace: S[i] becomes T[i]. Suffixes must be identical after the first char.
    # 2. Delete: S[i] is removed. S suffix from index 1 must match T suffix from index 0.
    # 3. Insert: T[i] is added. S suffix from index 0 must match T suffix from index 1.
    
    # We use a list of boolean conditions and 'any()' to determine if any are true.
    # S == T is the case for 0 operations.
    is_possible = any([
        S == T,
        # Replace: lengths are same, and suffixes after the first mismatch are identical
        len(s_suffix) == len(t_suffix) and s_suffix[1:] == t_suffix[1:],
        # Delete: S is one char longer, and S's suffix from 1 matches T's suffix from 0
        len(s_suffix) == len(t_suffix) + 1 and s_suffix[1:] == t_suffix,
        # Insert: T is one char longer, and S's suffix from 0 matches T's suffix from 1
        len(t_suffix) == len(s_suffix) + 1 and s_suffix == t_suffix[1:]
    ])

    # Output result
    sys.stdout.write("Yes\n" if is_possible else "No\n")

if __name__ == "__main__":
    solve()