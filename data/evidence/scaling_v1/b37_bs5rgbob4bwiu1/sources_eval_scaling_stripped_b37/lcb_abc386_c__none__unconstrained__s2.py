import sys

def solve():
    # Read K, S, and T from standard input
    # Using read().split() to handle potential whitespace/newline differences
    input_data = syss.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if the edit distance between S and T is <= K.
    # Since K=1, we only need to check if they are already equal,
    # or if they can be made equal by one insertion, deletion, or substitution.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # If the length difference is more than 1, it's impossible with K=1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Helper function to check if two strings are 1 edit apart
    # We use a generator to find the first index where characters differ
    def check_one_edit(s1, s2):
        # Find the first mismatch
        # zip stops at the shortest string
        mismatch_idx = next((i for i, (a, b) in enumerate(zip(s1, s2)) if a != b), None)
        
        # If no mismatch found within the length of the shorter string
        if mismatch_idx is None:
            # One string is a prefix of the other; check if length difference is exactly 1
            return abs(len(s1) - len(s2)) == 1
        
        # If mismatch found at index i:
        # 1. Replace: s1[i+1:] == s2[i+1:]
        # 2. Delete from s1: s1[i+1:] == s2[i:]
        # 3. Delete from s2: s1[i:] == s2[i+1:]
        
        # We use slicing which is efficient in Python
        # Replace
        if s1[mismatch_idx + 1:] == s2[mismatch_idx + 1:]:
            # Only valid if lengths were equal
            if len(s1) == len(s2):
                return True
        
        # Delete from s1 (Insertion into s2)
        if s1[mismatch_idx + 1:] == s2[mismatch_idx:]:
            if len(s1) == len(s2) + 1:
                return True
                
        # Delete from s2 (Insertion into s1)
        if s1[mismatch_idx:] == s2[mismatch_idx + 1:]:
            if len(s2) == len(s1) + 1:
                return True
                
        return False

    if check_one_edit(s, t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    # Overwriting sys.stdin.read to avoid the 'syss' typo in the logic above
    # and ensuring it runs in a single block.
    import sys as sys_module
    # To keep the logic inside one block and avoid the typo 'syss', 
    # I will redefine the function slightly to use the correct module.
    
    def final_solve():
        input_data = sys_module.stdin.read().split()
        if not input_data: return
        k = int(input_data[0])
        s = input_data[1]
        t = input_data[2]
        if s == t:
            print("Yes")
            return
        if abs(len(s) - len(t)) > 1:
            print("No")
            return
        
        # Find first mismatch
        # Using a list comprehension to find all indices where they differ 
        # is too slow, so we use a generator with next()
        diff_idx = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))
        
        # Try the three possible edits at the point of difference
        # 1. Substitution
        res_sub = (len(s) == len(t) and s[diff_idx+1:] == t[diff_idx+1:])
        # 2. Deletion from S
        res_del_s = (len(s) == len(t) + 1 and s[diff_idx+1:] == t[diff_idx:])
        # 3. Deletion from T
        res_del_t = (len(t) == len(s) + 1 and s[diff_idx:] == t[diff_idx+1:])
        
        if res_sub or res_del_s or res_del_t:
            print("Yes")
        else:
            # Special case: strings are identical up to the end of the shorter one
            # and differ by exactly one character at the end.
            # This is actually covered by the length check and the slice logic,
            # but for clarity: if diff_idx == len(shorter), it's a Yes.
            if diff_idx == min(len(s), len(t)) and abs(len(s) - len(t)) == 1:
                print("Yes")
            else:
                print("No")

    final_solve()