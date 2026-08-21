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
        
        # Read S from the second line
        s = sys.stdin.readline().strip()
    except ValueError:
        return

    # To handle duplicate characters in S and avoid counting the same 
    # permutation multiple times, we use a set of all unique permutations.
    # Given the constraints N <= 10, N! is at most 3,628,800, 
    # which is manageable in Python within a few seconds.
    
    # Generate all unique permutations of the string S
    # We use a set to automatically handle duplicates
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # Join the tuple into a string
        t = "".join(p)
        
        # Check if any substring of length K is a palindrome
        has_palindrome = False
        # There are (N - K + 1) substrings of length K
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                has_palindrome = True
                break
        
        # If no palindrome of length K was found, increment count
        if not has_palindrome:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()