import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if Edit Distance(S, T) <= 1
    # We find the first index where S and T differ.
    # Using a generator expression with next() to find the first mismatch index.
    # We zip S and T to compare characters side-by-side.
    
    # Find the first index i where s[i] != t[i]
    # We use a generator to find the first index of mismatch.
    # If no mismatch is found, i will be the length of the shorter string.
    i = next((idx for idx, (char_s, char_t) in enumerate(zip(s, t)) if char_s != char_t), 
             min(len(s), len(t)))

    # Suffixes after the first mismatch
    s_suffix = s[i:]
    t_suffix = t[i:]

    # There are three possibilities for 1 edit:
    # 1. Replace: s[i] is replaced by t[i]. Check if s[i+1:] == t[i+1:]
    # 2. Delete: s[i] is deleted. Check if s[i+1:] == t[i:]
    # 3. Insert: t[i] is inserted into s. Check if s[i:] == t[i+1:]
    
    # We use a list of booleans and 'any()' to check if any condition is met.
    # We must also handle the case where s == t (0 edits).
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Check if the edit distance is 1
    # We use slicing to check the three possible 1-edit scenarios.
    possible = any([
        # Replace
        (len(s_suffix) > 0 and len(t_suffix) > 0 and s_suffix[1:] == t_suffix[1:]),
        # Delete from S
        (len(s_suffix) > 0 and s_suffix[1:] == t_suffix),
        # Insert into S (Delete from T)
        (len(t_suffix) > 0 and s_suffix == t_suffix[1:])
    ])

    # Special case: if one string is exactly one character longer than the other 
    # and they are otherwise identical, the logic above covers it.
    # However, we must ensure we don't print Yes if the length difference > 1.
    
    if abs(len(s) - len(t)) <= 1 and possible:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()