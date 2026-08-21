import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem specifies K=1 for this sub-problem.
    # We need to check if edit distance between S and T is <= 1.
    
    # Find the first index where S and T differ
    # We use a generator expression with next() to find the first mismatch
    mismatch_idx = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))

    # Suffixes after the first mismatch
    s_suffix = s[mismatch_idx:]
    t_suffix = t[mismatch_idx:]

    # Case 1: S and T are already identical
    if s == t:
        print("Yes")
        return

    # Case 2: Try the three possible single-edit operations
    # 1. Replace: S[i] becomes T[i], so S[i+1:] must equal T[i+1:]
    # 2. Delete: Remove S[i], so S[i+1:] must equal T[i:]
    # 3. Insert: Add T[i] to S, so S[i:] must equal T[i+1:]
    
    # We use a list of boolean conditions and any() to check if any operation works
    possible = any([
        # Replace
        len(s_suffix) == len(t_suffix) and s_suffix[1:] == t_suffix[1:],
        # Delete from S
        len(s_suffix) == len(t_suffix) + 1 and s_suffix[1:] == t_suffix,
        # Insert into S (Delete from T)
        len(t_suffix) == len(s_suffix) + 1 and s_suffix == t_suffix[1:]
    ])

    print("Yes" if possible else "No")

if __name__ == "__main__":
    solve()