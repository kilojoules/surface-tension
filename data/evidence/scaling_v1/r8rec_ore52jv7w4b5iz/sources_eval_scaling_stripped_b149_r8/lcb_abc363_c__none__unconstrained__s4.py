import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Since N <= 10, N! is at most 3,628,800, which fits in memory/time
    # We use a set to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if any substring of length K is a palindrome
    def has_palindrome_k(t):
        # Check every substring of length K
        # t is a tuple of characters
        for i in range(n - k + 1):
            substring = t[i : i + k]
            # A substring is a palindrome if it equals its reverse
            if substring == substring[::-1]:
                return True
        return False

    # Use filter to keep only permutations that do NOT contain a palindrome of length K
    # The condition is 'not has_palindrome_k(p)'
    valid_perms = filter(lambda p: not has_palindrome_k(p), all_perms)

    # Count the number of valid permutations
    # We use a generator expression inside sum() to avoid creating a list
    result = sum(1 for _ in valid_perms)
    
    # Print the final result
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()