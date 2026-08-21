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
    # Since N is small (up to 10), we can use itertools.permutations 
    # and store them in a set to handle duplicate characters.
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # Convert tuple of characters back to a string
        t = "".join(p)
        
        # Check if any substring of length K is a palindrome
        has_palindrome = False
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                has_palindrome = True
                break
        
        # If no palindrome of length K is found, increment the counter
        if not has_palindrome:
            count += 1
            
    # Output the final result
    print(count)

if __name__ == "__main__":
    solve()