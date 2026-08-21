import sys

def solve():
    # Read K, S, and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance must be <= K. 
    # If the length difference is > K, it's impossible.
    if abs(n - m) > k:
        print("No")
        return

    # For K=1, we can check the three possible operations:
    # 1. S and T are already identical (0 operations)
    if s == t:
        print("Yes")
        return

    # 2. Replace one character: lengths must be equal
    # We check if they differ by exactly one character.
    if n == m:
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return
        else:
            print("No")
            return

    # 3. Delete one character from S (or insert into T): n = m + 1
    # 4. Insert one character into S (or delete from T): n = m - 1
    # Both are covered by checking if the shorter string is a subsequence of the longer one.
    if abs(n - m) == 1:
        longer = s if n > m else t
        shorter = t if n > m else s
        
        # Use a generator to find the first index where they differ
        # Then check if the remainder of the strings match after skipping one char in 'longer'
        # We find the first mismatch index 'i'
        # The only way to fix it in 1 op is if longer[i+1:] == shorter[i:]
        
        # Finding the first mismatch using a list comprehension to avoid loops
        # We find all indices where they differ and take the first one.
        # Since we can't use loops, we use zip and enumerate.
        mismatches = [i for i, (a, b) in enumerate(zip(longer, shorter)) if a != b]
        
        if not mismatches:
            # The shorter string is a prefix of the longer string
            print("Yes")
        else:
            i = mismatches[0]
            # Check if skipping the character at i in the longer string makes them equal
            if longer[i+1:] == shorter[i:]:
                print("Yes")
            else:
                print("No")
        return

    print("No")

if __name__ == "__main__":
    solve()