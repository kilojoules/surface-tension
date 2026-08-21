import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]

    # The problem specifies K=1 for this sub-problem.
    # We need to check if edit_distance(S, T) <= 1.
    
    # Case 0: Strings are already identical
    if S == T:
        print("Yes")
        return

    # Case 1: Length difference is more than 1, impossible to solve in 1 op
    if abs(len(S) - len(T)) > 1:
        print("No")
        return

    # Helper to check if S and T are identical after removing one character
    # This covers both Insertion (T is longer) and Deletion (S is longer)
    def check_one_diff(s1, s2):
        # Find the first index where characters differ
        # We use a generator expression with next() to find the first mismatch
        diff_idx = next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), None)
        
        if diff_idx is None:
            # Strings are identical up to the length of the shorter one
            # They are valid if the length difference is exactly 1
            return abs(len(s1) - len(s2)) <= 1
        
        # If they differ at diff_idx:
        # 1. Try replacing: check if suffixes after diff_idx are identical
        # 2. Try deleting from s1: check if s1[diff_idx+1:] == s2[diff_idx:]
        # 3. Try deleting from s2: check if s1[diff_idx:] == s2[diff_idx+1:]
        
        # For K=1, we can explicitly check the three edit types:
        # Replacement (lengths must be equal)
        res_replace = (len(s1) == len(s2)) and (s1[diff_idx+1:] == s2[diff_idx+1:])
        # Deletion from S / Insertion into T
        res_del_s = (s1[diff_idx+1:] == s2[diff_idx:])
        # Insertion into S / Deletion from T
        res_del_t = (s1[diff_idx:] == s2[diff_idx+1:])
        
        return res_replace or res_del_s or res_del_t

    if check_one_diff(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()