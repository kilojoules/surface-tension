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
    # We use a set of tuples to ensure uniqueness of permutations
    all_perms = set(permutations(s))

    # Helper function to check if a string (tuple) contains a palindrome of length K
    # A substring of length K starting at index i is a palindrome if 
    # it reads the same forwards and backwards.
    def contains_palindrome(p):
        # Check every possible starting position i for a substring of length K
        # The condition T_{i+j} = T_{i+K+1-j} is essentially checking if the 
    # substring is a palindrome.
        for i in range(n - k + 1):
            substring = p[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do NOT contain a palindrome of length K
    # We use a list comprehension to iterate through the set of unique permutations
    result = [p for p in all_perms if not contains_palindrome(p)]

    # Output the count of valid permutations
    print(len(result))

if __name__ == "__main__":
    solve()