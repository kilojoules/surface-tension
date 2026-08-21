import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # Function to find the length of the common prefix
    def get_prefix_len(s1, s2):
        # Using a generator expression with next() to find the first index where characters differ
        # We use range(min(len(s1), len(s2))) to avoid index out of bounds
        return next((i for i in range(min(len(s1), len(s2))) if s1[i] != s2[i]), min(len(s1), len(s2)))

    # Main logic to check if edit distance <= 1
    def check():
        # Case 0: Strings are already identical
        if s == t:
            return "Yes"
        
        # Find the first index where s and t differ
        i = get_prefix_len(s, t)
        
        # After the common prefix, we have suffixes s[i:] and t[i:]
        # To be within 1 edit distance, one of the following must be true:
        # 1. Replace: s[i+1:] == t[i+1:] (and lengths were compatible)
        # 2. Delete from S: s[i+1:] == t[i:]
        # 3. Insert into S: s[i:] == t[i+1:]
        
        # We use a list of boolean conditions and any() to check if any are true
        conditions = [
            # Replace: change s[i] to t[i]
            (i < len(s) and i < len(t) and s[i+1:] == t[i+1:]),
            # Delete: remove s[i]
            (i < len(s) and s[i+1:] == t[i:]),
            # Insert: add t[i] before s[i]
            (i < len(t) and s[i:] == t[i+1:])
        ]
        
        return "Yes" if any(conditions) else "No"

    print(check())

if __name__ == "__main__":
    solve()