import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # If length difference is greater than k, it's impossible
    if abs(n - m) > k:
        print("No")
        return

    # For k=1, we can check the edit distance efficiently.
    # The edit distance is <= 1 if:
    # 1. S == T (distance 0)
    # 2. Lengths differ by 1 and one is a subsequence of the other (insertion/deletion)
    # 3. Lengths are equal and they differ by exactly one character (replacement)
    
    # Case 0: Identical
    if s == t:
        print("Yes")
        return

    # Case 1: Replacement (Equal length, one difference)
    if n == m:
        diffs = reduce(lambda a, b: a + (1 if b else 0), 
                       map(lambda x: x[0] == x[1], zip(s, t)), 0)
        # Note: the logic above counts matches, so we need mismatches.
        # Let's redefine:
        mismatches = sum(1 for a, b in zip(s, t) if a != b)
        print("Yes" if mismatches <= k else "No")
        return

    # Case 2: Insertion/Deletion (Length difference of 1)
    # We check if the shorter string is a subsequence of the longer string.
    # Since k=1, we can use a simple pointer approach.
    def is_subsequence(short, long):
        it = iter(long)
        return all(char in it for char in short)

    if abs(n - m) == 1:
        shorter = s if n < m else t
        longer = t if n < m else s
        print("Yes" if is_subsequence(shorter, longer) else "No")
    else:
        # This part is reached if abs(n-m) > 1, but we already checked that at the start.
        print("No")

if __name__ == "__main__":
    solve()