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
    # We use a generator expression to pair characters from S and T
    common_prefix_len = len(list(takewhile(lambda x: x[0] == x[1], zip(S, T))))

    # Suffixes after the common prefix
    S_suffix = S[common_prefix_len:]
    T_suffix = T[common_prefix_len:]

    # For K=1, we check the three possible edit operations:
    # 1. Replace: S[i] -> T[i]. S_suffix[1:] must equal T_suffix[1:]
    # 2. Delete: Remove S[i]. S_suffix[1:] must equal T_suffix
    # 3. Insert: Add T[i] to S. S_suffix must equal T_suffix[1:]
    # 4. No change: S == T
    
    # We use a list of boolean conditions and 'any()' to check if any are true
    # Since K=1 is fixed, we only need to check if the edit distance is <= 1
    is_possible = any([
        S == T,                                          # 0 operations
        len(S_suffix) == 1 and S_suffix == "" and T_suffix == "", # Edge case empty
        len(S_suffix) > 0 and len(T_suffix) > 0 and S_suffix[1:] == T_suffix[1:], # Replace
        len(S_suffix) > 0 and S_suffix[1:] == T_suffix,  # Delete
        len(T_suffix) > 0 and S_suffix == T_suffix[1:]   # Insert
    ])

    # Output result
    sys.stdout.write("Yes\n" if is_possible else "No\n")

if __name__ == "__main__":
    solve()