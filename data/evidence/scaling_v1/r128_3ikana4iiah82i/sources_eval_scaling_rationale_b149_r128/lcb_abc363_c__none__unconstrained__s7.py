import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set of tuples to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    def contains_palindrome(p):
        # p is a tuple of characters
        # We check every slice of length k
        return any(
            p[i : i + k] == p[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # Count permutations that do NOT contain a palindrome of length K
    # sum() treats True as 1 and False as 0
    result = sum(
        1 for p in all_perms 
        if not contains_palindrome(p)
    )

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()