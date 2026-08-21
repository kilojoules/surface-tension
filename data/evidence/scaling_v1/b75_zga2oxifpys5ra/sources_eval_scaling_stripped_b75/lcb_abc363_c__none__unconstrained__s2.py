import sys
from itertools import permutations

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
    all_permutations = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there is some i such that
    # T[i:i+K] == T[i:i+K][::-1]
    def has_palindrome_k(p):
        # p is a tuple of characters
        # We check every substring of length K
        for i in range(n - k + 1):
            substring = p[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do not contain a palindrome of length K
    # and count them
    result = sum(1 for p in all_permutations if not has_palindrome_k(p))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()