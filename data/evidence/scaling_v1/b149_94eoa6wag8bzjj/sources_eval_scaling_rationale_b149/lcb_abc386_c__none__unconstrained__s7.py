import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Since K=1, we can use a functional approach to find the first mismatch
    # and check the three possible edit scenarios.
    
    # Find the length of the common prefix
    # We use a generator expression with next() to find the first index where s[i] != t[i]
    # without using a for loop.
    def get_first_mismatch(s, t):
        # zip stops at the shortest string
        # next() finds the first index where the condition is true
        # We use a default value (max length) if no mismatch is found
        return next((i for i, (a, b) in enumerate(zip(s, t)) if a != b), min(len(s), len(t)))

    mismatch_idx = get_first_mismatch(s, t)

    # If the strings are identical up to the length of the shorter one
    # we check if the length difference is <= K
    if mismatch_idx == min(len(s), len(t)):
        print("Yes" if abs(len(s) - len(t)) <= k else "No")
        return

    # If there is a mismatch, we check the 3 possible operations:
    # 1. Replace: s[idx+1:] == t[idx+1:]
    # 2. Delete from S: s[idx+1:] == t[idx:]
    # 3. Insert into S: s[idx:] == t[idx+1:]
    
    # We use a list of booleans and the any() function to avoid if/else blocks and loops
    results = [
        s[mismatch_idx + 1:] == t[mismatch_idx + 1:], # Replace
        s[mismatch_idx + 1:] == t[mismatch_idx:],     # Delete
        s[mismatch_idx:] == t[mismatch_idx + 1:]      # Insert
    ]

    print("Yes" if any(results) else "No")

if __name__ == "__main__":
    solve()