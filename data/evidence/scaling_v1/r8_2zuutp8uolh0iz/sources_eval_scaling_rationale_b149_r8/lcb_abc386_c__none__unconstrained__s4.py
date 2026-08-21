import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Since K=1, we check if edit distance is 0 or 1.
    # Case 0: Strings are identical
    is_equal = (s == t)

    # Case 1: Substitution (lengths must be equal)
    # Check if they differ by exactly one character
    is_sub = (len(s) == len(t) and 
               sum(1 for a, b in zip(s, t) if a != b) <= 1)

    # Case 2: Deletion from S (or insertion into T)
    # S is one character longer than T
    # We look for the first mismatch and check if the rest of S (from i+1) matches T (from i)
    def check_single_diff(s1, s2):
        # Find first index where characters differ
        # Using a generator to find the first mismatch index
        mismatches = [i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]]
        
        # If no mismatch in the prefix, it's a diff at the very end
        if not mismatches:
            return abs(len(s1) - len(s2)) <= 1
        
        idx = mismatches[0]
        # For deletion from s1: s1[idx+1:] must match s2[idx:]
        # For insertion into s1: s1[idx:] must match s2[idx+1:]
        return (s1[idx+1:] == s2[idx:]) or (s1[idx:] == s2[idx+1:])

    # The logic for K=1 can be simplified:
    # 1. Length difference > 1 -> No
    # 2. Lengths equal -> check if 0 or 1 chars differ
    # 3. Length difference == 1 -> check if removing one char makes them equal
    
    # Using a more functional approach to handle the length-diff-1 case:
    # We find the first mismatch and check the remaining suffixes.
    
    # To avoid loops, we use a helper to find the first mismatch index
    # We can use next() with a generator
    def get_first_mismatch(s1, s2):
        return next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))

    # Logic for length difference of 1
    def check_len_diff_one(s1, s2):
        # Ensure s1 is the longer string
        long, short = (s1, s2) if len(s1) > len(s2) else (s2, s1)
        if abs(len(s1) - len(s2)) != 1:
            return False
        idx = get_first_mismatch(long, short)
        return long[idx+1:] == short[idx:]

    # Final result
    result = is_equal or is_sub or check_len_diff_one(s, t)
    
    sys.stdout.write("Yes\n" if result else "No\n")

if __name__ == "__main__":
    solve()