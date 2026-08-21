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

    # The constraints N <= 10 allow us to generate all permutations
    # of the string S and check each one.
    # Using set(permutations(s)) automatically handles duplicate characters
    # in S, ensuring we only count unique permutations.
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # Convert the tuple of characters back into a string
        t = "".join(p)
        
        # Check if the string contains any palindrome of length K
        contains_palindrome = False
        # Iterate through all possible substrings of length K
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                contains_palindrome = True
                break
        
        # If no palindrome of length K was found, increment the count
        if not contains_palindrome:
            count += 1
            
    # Print the final result to standard output
    print(count)

if __name__ == "__main__":
    solve()