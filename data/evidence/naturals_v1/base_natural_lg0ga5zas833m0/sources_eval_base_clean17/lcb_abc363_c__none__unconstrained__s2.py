import sys
from itertools import permutations

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

def solve():
    # Read N and K from the first line of input
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, k = map(int, line1)
        # Read the string S
        s = sys.stdin.readline().strip()
    except ValueError:
        return

    # Generate all unique permutations of the string S
    # Since N is small (up to 10), we can iterate through all permutations
    # and use a set to keep track of unique strings.
    # However, using itertools.permutations on the characters and then 
    # putting them into a set is feasible for N=10 (10! = 3,628,800).
    
    all_perms = set(permutations(s))
    count = 0
    
    for p in all_perms:
        # Convert the tuple of characters back to a string
        t = "".join(p)
        
        # Check if the string contains any palindrome of length K as a substring
        has_palindrome = False
        # There are (N - K + 1) possible starting positions for a substring of length K
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                has_palindrome = True
                break
        
        if not has_palindrome:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()