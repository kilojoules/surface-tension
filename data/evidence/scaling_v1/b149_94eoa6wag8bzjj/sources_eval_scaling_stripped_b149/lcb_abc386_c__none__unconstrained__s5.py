import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if len(input_data) < 3:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem is to check if Edit Distance(S, T) <= 1
    # 1. If lengths differ by more than 1, it's impossible
    if abs(len(s) - len(t)) > 1:
        sys.stdout.write("No\n")
        return

    # 2. If lengths are equal, they must differ by at most 1 character
    if len(s) == len(t):
        # Count mismatches using a generator expression
        diff_count = sum(1 for a, b in zip(s, t) if a != b)
        sys.stdout.write("Yes\n" if diff_count <= 1 else "No\n")
        return

    # 3. If lengths differ by 1, the shorter must be a subsequence of the longer
    shorter, longer = (s, t) if len(s) < len(t) else (t, s)
    
    # Find the first index where characters differ
    # We use a list comprehension to find all indices of differences 
    # but we only need the first one to check the shift.
    # Since we cannot use loops, we find the first mismatch by 
    # comparing prefixes and suffixes or using a helper.
    
    # We can find the first mismatch index by checking which prefix of 'shorter' 
    # matches the prefix of 'longer'.
    # However, a simpler way to check if 'shorter' is 'longer' minus one char:
    # There must exist some i such that longer[:i] + longer[i+1:] == shorter
    # But we can't loop i. 
    # Correct approach: find the first index i where s[i] != t[i].
    # Then check if s[i:] == t[i+1:].
    
    # To find the first mismatch index without a loop:
    # We can use a generator with next()
    try:
        first_mismatch = next(i for i in range(len(shorter)) if shorter[i] != longer[i])
        result = (shorter[first_mismatch:] == longer[first_mismatch + 1:])
    except StopIteration:
        # Shorter is a prefix of longer
        result = True

    sys.stdout.write("Yes\n" if result else "No\n")

if __name__ == "__main__":
    solve()