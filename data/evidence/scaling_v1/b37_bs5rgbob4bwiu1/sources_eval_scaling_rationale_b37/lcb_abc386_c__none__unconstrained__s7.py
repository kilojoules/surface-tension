import sys

def solve():
    # Read input and strip whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Find the length of the common prefix
    # We use a generator expression with next() to find the first index where s and t differ
    # zip() pairs characters, enumerate() tracks the index
    diff_idx = next((i for i, (a, b) in enumerate(zip(s, t)) if a != b), min(len(s), len(t)))

    # Suffixes after the common prefix
    s_suffix = s[diff_idx:]
    t_suffix = t[diff_idx:]

    # Case 1: Strings are already identical
    if not s_suffix and not t_suffix:
        print("Yes")
        return

    # Case 2: Try the three possible operations for K=1
    # 1. Replace: s[i] becomes t[i]. Check if s[i+1:] == t[i+1:]
    # 2. Delete: s[i] is removed. Check if s[i+1:] == t[i:]
    # 3. Insert: t[i] is added to s. Check if s[i:] == t[i+1:]
    
    # We use a list of boolean checks and the any() function to avoid if/else blocks
    # and explicit loops.
    possibilities = [
        # Replace
        (len(s_suffix) == len(t_suffix) and s_suffix[1:] == t_suffix[1:]),
        # Delete from S
        (len(s_suffix) == len(t_suffix) + 1 and s_suffix[1:] == t_suffix),
        # Insert into S (Delete from T)
        (len(t_suffix) == len(s_suffix) + 1 and s_suffix == t_suffix[1:])
    ]

    if any(possibilities):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()