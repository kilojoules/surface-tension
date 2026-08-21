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
    all_perms = set(permutations(s))

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists an i
    # such that T[i:i+K] == T[i:i+K][::-1]
    def has_palindrome_k(t):
        # Check every substring of length K
        for i in range(n - k + 1):
            substring = t[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We map the tuple from permutations back to a string for the check
    result = sum(1 for p in all_perms if not has_palindrome_k("".join(p)))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()