import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if edit distance <= 1
    # We find the first index where s and t differ
    # Using a generator expression and next() to find the first mismatch
    # We zip s and t to compare them character by character
    mismatch_idx = next(
        (i for i, (a, b) in enumerate(zip(s, t)) if a != b), 
        min(len(s), len(t))
    )

    # Slices of the strings from the first mismatch onwards
    s_rem = s[mismatch_idx:]
    t_rem = t[mismatch_idx:]

    # Check the three possible operations:
    # 1. Replace: s[i] becomes t[i]. Check if s[i+1:] == t[i+1:]
    # 2. Delete: s[i] is removed. Check if s[i+1:] == t[i:]
    # 3. Insert: t[i] is added to s. Check if s[i:] == t[i+1:]
    
    # We use a list of booleans and 'any()' to avoid explicit if/else loops
    # Note: slicing handles out-of-bounds indices gracefully in Python
    possible = any([
        # Replace or identical
        (s_rem[1:] == t_rem[1:]),
        # Delete from S
        (s_rem[1:] == t_rem),
        # Insert into S (Delete from T)
        (s_rem == t_rem[1:])
    ])

    # Special case: if the strings are identical, the logic above still holds
    # because s_rem and t_rem would be empty or single chars.
    # However, we must ensure we don't count a "replacement" if the strings 
    # are already identical and we are at the end of the strings.
    # Actually, the logic (s_rem[1:] == t_rem[1:]) covers s==t perfectly.

    print("Yes" if possible else "No")

if __name__ == "__main__":
    solve()