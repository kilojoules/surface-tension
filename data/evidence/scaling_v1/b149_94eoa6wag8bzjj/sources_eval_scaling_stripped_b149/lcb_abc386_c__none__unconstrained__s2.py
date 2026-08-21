import sys
from itertools import groupby

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if the Edit Distance (Levenshtein distance) is <= K.
    # For K=1, we can check this by comparing the strings.
    
    # Case 0: Strings are identical
    if s == t:
        print("Yes")
        return

    # If K=0 and strings aren't identical, it's impossible.
    # However, the constraint says K=1 for this subproblem.
    if k == 0:
        print("No")
        return

    # For K=1, the distance is <= 1 if:
    # 1. Length difference is > 1: Impossible
    len_s, len_t = len(s), len(t)
    if abs(len_s - len_t) > 1:
        print("No")
        return

    # We can check if distance is 1 by finding the first mismatch
    # and then checking the remaining suffixes based on the operation.
    
    # Find the first index where s and t differ
    # Using a generator to find the first mismatch index
    mismatch_idx = next((i for i in range(min(len_s, len_t)) if s[i] != t[i]), min(len_s, len_t))

    # Suffixes after the first mismatch
    s_suffix = s[mismatch_idx:]
    t_suffix = t[mismatch_idx:]

    # Possible operations for K=1:
    # 1. Replace: s[i] becomes t[i]. Check if s[i+1:] == t[i+1:]
    # 2. Delete from S: s[i] is removed. Check if s[i+1:] == t[i:]
    # 3. Insert into S: t[i] is added. Check if s[i:] == t[i+1:]
    
    # We use a list of boolean checks and 'any()' to determine if any operation works.
    # We handle slicing carefully to avoid index errors.
    
    result = any([
        # Replace
        (len_s == len_t and s[mismatch_idx+1:] == t[mismatch_idx+1:]),
        # Delete from S
        (len_s == len_t + 1 and s[mismatch_idx+1:] == t[mismatch_idx:]),
        # Insert into S (Delete from T)
        (len_t == len_s + 1 and s[mismatch_idx:] == t[mismatch_idx+1:])
    ])

    if result:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()