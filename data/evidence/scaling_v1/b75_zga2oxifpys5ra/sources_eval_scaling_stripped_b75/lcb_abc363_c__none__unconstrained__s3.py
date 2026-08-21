import itertools
import sys

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
    perms = set(itertools.permutations(s))

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there is some i
    # such that T[i:i+K] == T[i:i+K][::-1]
    def contains_palindrome(tup):
        # We check every substring of length K
        for i in range(n - k + 1):
            substring = tup[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in perms if not contains_palindrome(p))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()