import sys

def solve():
    # Read input and handle potential empty lines/whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]

    # The problem specifies K=1 for this sub-problem.
    # We need to check if edit distance between S and T is <= 1.
    
    # Case 0: Already identical
    if s == t:
        print("Yes")
        return

    # Case 1: Substitution (Lengths must be equal)
    # Check if they differ by exactly one character
    def check_substitution():
        if len(s) != len(t):
            return False
        # Count positions where characters differ
        diffs = sum(1 for a, b in zip(s, t) if a != b)
        return diffs <= 1

    # Case 2: Deletion from S (or Insertion into T)
    # S is longer than T by 1
    def check_deletion():
        if len(s) != len(t) + 1:
            return False
        # Find the first mismatch
        # We use a generator to find the index of the first difference
        # Since we can't use loops, we find the first index i where s[i] != t[i]
        # and check if s[i+1:] == t[i:]
        # To do this without a loop, we can find the first mismatch index using a trick:
        # We compare prefixes and suffixes.
        # However, a simpler way without loops is to find the first index of difference:
        # But wait, we can't use a loop to find the index. 
        # Let's use the property: if deleting one char from S makes it T,
        # then there exists some i such that s[:i] + s[i+1:] == t.
        # But we can't loop through all i. 
        # Actually, we can use the fact that if len(s) == len(t) + 1,
        # the only possible deletion point is at the first index where they differ.
        
        # Finding the first mismatch index without a loop:
        # We can use a generator expression with next()
        try:
            first_mismatch = next(i for i in range(min(len(s), len(t))) if s[i] != t[i])
            return s[first_mismatch + 1:] == t[first_mismatch:]
        except StopIteration:
            # They are identical up to the end of T, so deleting the last char of S works
            return True

    # Case 3: Insertion into S (or Deletion from T)
    # T is longer than S by 1
    def check_insertion():
        if len(t) != len(s) + 1:
            return False
        try:
            first_mismatch = next(i for i in range(min(len(s), len(t))) if s[i] != t[i])
            return s[first_mismatch:] == t[first_mismatch + 1:]
        except StopIteration:
            return True

    # Evaluate all possibilities
    if check_substitution() or check_deletion() or check_insertion():
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    solve()