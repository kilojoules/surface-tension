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
    def contains_palindrome(t):
        # Check every substring of length K
        # t is a tuple of characters
        for i in range(n - k + 1):
            substring = t[i : i + k]
            # A substring is a palindrome if it reads the same forwards and backwards
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() to count valid permutations
    result = sum(1 for p in all_perms if not contains_palindrome(p))

    # Print the final count
    print(result)

if __name__ == "__main__":
    solve()