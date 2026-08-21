import sys
from itertools import permutations

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

def solve():
    # Read N and K from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, k = map(int, line1)
        # Read string S from the second line
        s = sys.stdin.readline().strip()
    except ValueError:
        return

    # Generate all unique permutations of the string S
    # Since N is small (up to 10), we can iterate through all permutations
    # and use a set to handle duplicates.
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # Convert tuple to string
        t = "".join(p)
        
        contains_palindrome = False
        # Check every substring of length K
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                contains_palindrome = True
                break
        
        if not contains_palindrome:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()