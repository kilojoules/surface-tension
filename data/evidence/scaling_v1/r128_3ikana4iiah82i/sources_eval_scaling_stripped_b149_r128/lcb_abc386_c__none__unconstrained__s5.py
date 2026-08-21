import sys
from itertools import groupby

def solve():
    # Read K, S, T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # The problem is to check if the edit distance between S and T is <= K.
    # For K=1, we can check this by comparing the strings and finding the first mismatch.
    
    # We use a helper to find the first index where S and T differ.
    # Since we cannot use loops, we use a generator expression and next().
    def get_first_diff(s, t):
        return next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))

    # Find the first point of divergence
    idx = get_first_diff(S, T)
    
    # Suffixes starting from the first difference
    s_suffix = S[idx:]
    t_suffix = T[idx:]
    
    # For K=1, there are three possibilities to make S identical to T:
    # 1. Replace: Change S[idx] to T[idx]. Check if S[idx+1:] == T[idx+1:]
    # 2. Delete: Remove S[idx]. Check if S[idx+1:] == T[idx:]
    # 3. Insert: Add T[idx] to S. Check if S[idx:] == T[idx+1:]
    
    # We use a list comprehension to check these conditions and 'any()' to return True/False.
    # We also handle the case where S == T (which is allowed since 0 <= operations <= K).
    
    result = (S == T) or any([
        # Replace
        (len(s_suffix) > 0 and len(t_suffix) > 0 and S[idx+1:] == T[idx+1:]),
        # Delete
        (len(s_suffix) > 0 and S[idx+1:] == T[idx:]),
        # Insert
        (len(t_suffix) > 0 and S[idx:] == T[idx+1:])
    ])
    
    print("Yes" if result else "No")

if __name__ == "__main__":
    solve()