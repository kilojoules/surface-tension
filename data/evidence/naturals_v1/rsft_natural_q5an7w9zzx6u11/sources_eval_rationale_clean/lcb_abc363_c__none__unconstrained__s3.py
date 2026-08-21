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
    # Using a set to automatically remove duplicates caused by identical characters
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string contains a palindrome of length K if any substring of length K is equal to its reverse
    # We use a generator expression inside any() to check all possible starting positions i
    def contains_palindrome(p):
        return any(
            p[i : i + k] == p[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # We need to count permutations that DO NOT contain a palindrome of length K
    # We use a generator expression inside sum() to count valid permutations
    # The condition is 'not contains_palindrome(p)'
    # Note: p is a tuple from itertools.permutations, and slicing/reversing works on tuples
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1]
            for i in range(n - k + 1)
        )
    )

    # Print the final count to standard output
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()