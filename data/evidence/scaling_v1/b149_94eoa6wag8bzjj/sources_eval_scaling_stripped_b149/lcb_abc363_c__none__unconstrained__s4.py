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
    # We use a set of tuples to ensure uniqueness since S may have duplicate characters
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if any substring of length K is equal to its reverse
    def contains_k_palindrome(t):
        # Check every substring of length K
        # t[i : i+k] is the substring, [::-1] reverses it
        return any(t[i : i+k] == t[i : i+k][::-1] for i in range(n - k + 1))

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not contains_k_palindrome(p))

    # Print the final count to standard output
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()