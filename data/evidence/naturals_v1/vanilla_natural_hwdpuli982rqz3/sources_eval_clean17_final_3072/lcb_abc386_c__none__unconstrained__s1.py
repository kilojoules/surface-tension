import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # If the length difference is more than K, it's impossible
    if abs(n - m) > k:
        print("No")
        return

    # Find the first index where s and t differ
    # We use a list comprehension to find all indices where they differ
    # and then take the first one.
    diff_indices = [i for i in range(min(n, m)) if s[i] != t[i]]
    
    # If they are identical up to the length of the shorter string
    # The difference is just the trailing characters of the longer string.
    first_diff = diff_indices[0] if diff_indices else min(n, m)
    
    # Slices from the first difference to the end
    s_rem = s[first_diff:]
    t_rem = t[first_diff:]
    
    # Possible operations for K=1:
    # 1. Replace: s_rem[0] is replaced by t_rem[0], then s_rem[1:] must equal t_rem[1:]
    # 2. Delete: s_rem[0] is deleted, then s_rem[1:] must equal t_rem
    # 3. Insert: t_rem[0] is inserted into s, then s_rem must equal t_rem[1:]
    
    # We check if any of these conditions are met.
    # Note: if s_rem or t_rem are empty, some slices might be empty, which is handled by Python.
    
    can_convert = (
        # Case 0: Already equal (handled by the logic below, but explicitly: if s==t)
        s == t or
        # Case 1: Replace
        (len(s_rem) > 0 and len(t_rem) > 0 and s_rem[1:] == t_rem[1:]) or
        # Case 2: Delete from S
        (len(s_rem) > 0 and s_rem[1:] == t_rem) or
        # Case 3: Insert into S (Delete from T)
        (len(t_rem) > 0 and s_rem == t_rem[1:])
    )
    
    if can_convert:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()