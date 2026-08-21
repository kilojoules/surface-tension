import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # K is the first element, S the second, T the third
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    # Case 0: Strings are already identical
    if s == t:
        print("Yes")
        return
    
    # Case 1: K is 0 but strings are different
    if k == 0:
        print("No")
        return

    # For K=1, we check if edit distance is exactly 1
    # 1. Length difference must be at most 1
    if abs(len(s) - len(t)) > 1:
        print("No")
        return

    # 2. If lengths are equal, check for exactly one substitution
    if len(s) == len(t):
        # Use a generator to count mismatches without a loop
        diff_count = sum(1 for a, b in zip(s, t) if a != b)
        print("Yes" if diff_count <= k else "No")
        return

    # 3. If lengths differ by 1, check for one insertion/deletion
    # Identify shorter and longer strings
    shorter, longer = (s, t) if len(s) < len(t) else (t, s)
    
    # Find the first index where characters differ
    # We use a generator expression with next() to find the first mismatch index
    first_diff = next((i for i in range(len(shorter)) if shorter[i] != longer[i]), len(shorter))
    
    # If we skip the character at first_diff in the longer string, 
    # the remaining suffixes must be identical.
    if shorter[first_diff:] == longer[first_diff + 1:]:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()