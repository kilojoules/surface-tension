import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read N and K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set of tuples to ensure uniqueness since S may have duplicate characters
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there's a substring of length K
    # that is equal to its own reverse.
    def contains_palindrome(p):
        # Check every substring of length K
        # The condition T_{i+j} = T_{i+K+1-j} is essentially the definition of a palindrome
        for i in range(n - k + 1):
            substring = p[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Use map to apply the check to all permutations
    # Then use reduce to count how many do NOT contain the palindrome
    # We use a list comprehension inside sum() for counting
    result = sum(1 for p in all_perms if not contains_palindrome(p))
    
    print(result)

if __name__ == "__main__":
    solve()