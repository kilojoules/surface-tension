import sys

def solve():
    # Read input and split into parts
    input_data = sys.stdin.read().split()
    if len(input_data) < 3:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # If length difference is more than k, it's impossible
    if abs(n - m) > k:
        print("No")
        return

    # For k=1, we check if s == t or if they are 1 edit apart.
    # We use a pointer-based approach to find the first mismatch.
    
    # Find first mismatch from left
    # Using a generator to find the first index where s[i] != t[i]
    # We can only compare up to the length of the shorter string.
    min_len = min(n, m)
    
    # Find the first index of mismatch
    # We use next() with a default value to avoid loops
    first_mismatch = next((i for i in range(min_len) if s[i] != t[i]), min_len)
    
    # Slices the strings from the first mismatch
    s_suff = s[first_mismatch:]
    t_suff = t[first_mismatch:]
    
    # Possible scenarios for k=1:
    # 1. Strings are identical
    # 2. One character replacement: len(s_suff) == len(t_suff) and s_suff[1:] == t_suff[1:]
    # 3. One character deletion from S: len(s_suff) == len(t_suff) + 1 and s_suff[1:] == t_suff
    # 4. One character insertion into S: len(s_suff) == len(t_suff) - 1 and s_suff == t_suff[1:]
    
    # We check if any of these are true. 
    # Note: if first_mismatch == min_len, one string is a prefix of the other.
    
    results = [
        s == t,
        len(s_suff) == len(t_suff) and s_suff[1:] == t_suff[1:],
        len(s_suff) == len(t_suff) + 1 and s_suff[1:] == t_suff,
        len(s_suff) == len(t_suff) - 1 and s_suff == t_suff[1:]
    ]
    
    if any(results):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()