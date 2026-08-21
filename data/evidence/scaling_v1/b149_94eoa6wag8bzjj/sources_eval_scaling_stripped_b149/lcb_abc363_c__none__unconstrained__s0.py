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
    # We use a set to ensure we only count distinct permutations
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there is some i such that
    # T[i:i+k] == T[i:i+k][::-1]
    def contains_palindrome(p):
        # p is a tuple of characters
        # We check every substring of length k
        for i in range(n - k + 1):
            substring = p[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do NOT contain a palindrome of length K
    # and count them.
    result = sum(1 for p in all_perms if not contains_palindrome(p))
    
    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()