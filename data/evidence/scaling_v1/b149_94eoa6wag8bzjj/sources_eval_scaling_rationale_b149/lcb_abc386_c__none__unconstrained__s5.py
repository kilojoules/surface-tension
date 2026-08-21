import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # If strings are identical, it's always possible (0 operations)
    if s == t:
        print("Yes")
        return

    # Since K=1, we can only afford one edit.
    # First, check if the length difference is greater than 1.
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # Find the first index where s and t differ
    # Use a generator to find the first mismatch
    first_diff = next((i for i in range(min(len(s), len(t))) if s[i] != t[i]), min(len(s), len(t)))

    # Find the last index where s and t differ by comparing reversed strings
    # This allows us to isolate the "middle" part that needs fixing
    s_rev, t_rev = s[::-1], t[::-1]
    last_diff_rev = next((i for i in range(min(len(s_rev), len(t_rev))) if s_rev[i] != t_rev[i]), min(len(s_rev), len(t_rev)))
    
    # Calculate the actual index from the end
    # last_diff_rev is the distance from the end
    
    # To check if 1 operation suffices, we check the three possible operations:
    # 1. Replace: s[first_diff] is changed to t[first_diff]
    #    Possible if len(s) == len(t) and s[first_diff+1:] == t[first_diff+1:]
    # 2. Delete from S: s[first_diff] is removed
    #    Possible if len(s) == len(t) + 1 and s[first_diff+1:] == t[first_diff:]
    # 3. Insert into S: t[first_diff] is inserted into s
    #    Possible if len(t) == len(s) + 1 and s[first_diff:] == t[first_diff+1:]

    # We can use a list of booleans and the 'any' function to avoid if/else blocks
    checks = [
        (len(s) == len(t) and s[first_diff+1:] == t[first_diff+1:]),
        (len(s) == len(t) + 1 and s[first_diff+1:] == t[first_diff:]),
        (len(t) == len(s) + 1 and s[first_diff:] == t[first_diff+1:])
    ]

    if any(checks):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()