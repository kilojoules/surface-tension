import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem asks if edit distance <= K. For K=1, we can check this efficiently.
    # First, check if they are already identical
    if s == t:
        print("Yes")
        return

    # Find the first index where S and T differ
    # We use a list comprehension to find all indices where they differ, 
    # then take the min of those indices.
    # To avoid loops, we use map and next with a generator.
    
    # Find the length of the shorter string to avoid index out of bounds
    min_len = min(len(s), len(t))
    
    # Find the first index i where s[i] != t[i]
    # next() allows us to find the first occurrence without a for loop
    diff_idx = next((i for i in range(min_len) if s[i] != t[i]), min_len)

    # There are three possibilities to fix the string in 1 operation:
    # 1. Replace: s[diff_idx] is changed to t[diff_idx]. 
    #    Check if s[diff_idx+1:] == t[diff_idx+1:]
    # 2. Delete from S: s[diff_idx] is removed.
    #    Check if s[diff_idx+1:] == t[diff_idx:]
    # 3. Insert into S: t[diff_idx] is added before s[diff_idx].
    #    Check if s[diff_idx:] == t[diff_idx+1:]

    # We check these conditions using a list of booleans and the any() function
    results = [
        # Replace
        (len(s) == len(t) and s[diff_idx+1:] == t[diff_idx+1:]),
        # Delete
        (len(s) == len(t) + 1 and s[diff_idx+1:] == t[diff_idx:]),
        # Insert
        (len(s) == len(t) - 1 and s[diff_idx:] == t[diff_idx+1:])
    ]

    if any(results):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()