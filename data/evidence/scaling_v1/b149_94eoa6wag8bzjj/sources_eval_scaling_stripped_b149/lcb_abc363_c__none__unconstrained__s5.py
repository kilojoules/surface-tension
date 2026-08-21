import sys
from itertools import permutations

def solve():
    # Read all input at once and split into a list of strings
    data = sys.stdin.read().split()
    if not data:
        return
    
    # N: length of string, K: length of palindrome to avoid
    n = int(data[0])
    k = int(data[1])
    s = data[2]

    # Generate all unique permutations of the characters in S
    # Using a set of tuples to automatically handle duplicate characters
    unique_perms = set(permutations(s))

    # Helper function to check if a permutation contains a palindrome of length K
    # A string is a palindrome if it reads the same forwards and backwards
    def has_k_palindrome(t):
        # Check every substring of length k
        for i in range(n - k + 1):
            # Extract substring of length k starting at index i
            sub = t[i : i + k]
            # Check if the substring is equal to its reverse
            if sub == sub[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() to avoid creating a list in memory
    ans = sum(1 for p in unique_perms if not has_k_palindrome(p))

    # Output the result to standard output
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()