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
    # Using a set of permutations handles duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there's an index i
    # such that the substring T[i : i+K] is equal to its reverse.
    def has_palindrome_k(p):
        # p is a tuple of characters
        # We check every possible substring of length K
        for i in range(n - k + 1):
            substring = p[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() to count the valid permutations
    result = sum(1 for p in all_perms if not has_palindrome_k(p))

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()