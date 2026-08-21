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
    # We use a set of tuples to handle duplicate characters in S
    all_perms = set(permutations(s))

    def contains_palindrome_of_length_k(t, n, k):
        # Check every substring of length k
        # A substring is a palindrome if it equals its reverse
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(
        1 for p in all_perms 
        if not contains_palindrome_of_length_k(p, n, k)
    )

    # Print the final count to standard output
    sys.stdout.write(str(result) + '\n')

# Execute the solve function
if __name__ == "__main__":
    solve()