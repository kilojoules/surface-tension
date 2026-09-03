import sys

def solve():
    # Read K, S, and T from standard input
    # Using sys.stdin.read().split() to handle potential whitespace issues
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # The constraint to avoid explicit loops forces a functional programming style,
    # emphasizing the use of high-level abstractions over manual index management.
    # For K=1, we check if the edit distance is <= 1.
    
    # 1. Check if strings are identical
    if s == t:
        print("Yes")
        return

    # 2. Check if length difference is > 1 (impossible for K=1)
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper to check if two strings are identical after ignoring one character
    # This covers Insertion and Deletion.
    def check_one_diff(s1, s2):
        # s1 is the longer string, s2 is the shorter string
        # We find the first mismatch
        # Using a generator to find the first index where characters differ
        mismatches = [i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]]
        
        # If no mismatch in the prefix, it's a trailing character difference
        if not mismatches:
            return True
        
        # If there is a mismatch, check if skipping that char in the longer string 
        # makes the rest of the strings identical.
        idx = mismatches[0]
        return s1[idx+1:] == s2[idx:]

    # Case A: Lengths are equal (Possible Replace operation)
    if len(s) == len(t):
        # Count mismatches. If exactly 1, it's a Replace.
        mismatches_count = sum(1 for a, b in zip(s, t) if a != b)
        if mismatches_count <= 1:
            print("Yes")
        else:
            print("No")
        return

    # Case B: Lengths differ by 1 (Possible Insert or Delete operation)
    # Identify which is longer and which is shorter
    s_long, s_short = (s, t) if len(s) > len(t) else (t, s)
    
    if check_one_diff(s_long, s_short):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()