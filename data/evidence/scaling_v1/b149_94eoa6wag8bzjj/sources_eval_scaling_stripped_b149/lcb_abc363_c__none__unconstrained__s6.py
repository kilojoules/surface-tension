import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there's an index i
    # such that T[i:i+k] == T[i:i+k][::-1]
    def contains_palindrome(p):
        # We check every substring of length K
        # The range is from 0 to N-K inclusive
        return any(
            p[i : i + k] == p[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # Use map to apply the check to all permutations
    # Use reduce to count how many do NOT contain the palindrome
    # result = sum(1 for p in all_perms if not contains_palindrome(p))
    # Implementing the count using reduce as a functional approach
    result = reduce(lambda acc, p: acc + (0 if contains_palindrome(p) else 1), all_perms, 0)

    print(result)

if __name__ == "__main__":
    solve()