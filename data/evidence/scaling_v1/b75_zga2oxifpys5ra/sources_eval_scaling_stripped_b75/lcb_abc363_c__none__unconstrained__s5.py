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
    # We use a set to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+K] == T[i:i+K][::-1]
    def has_palindrome_k(tup):
        # We check every substring of length K
        for i in range(n - k + 1):
            substring = tup[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not has_palindrome_k(p))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()