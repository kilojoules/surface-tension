import sys
from itertools import permutations, combinations_with_replacement
from collections import Counter

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # Helper to check if a string contains a palindrome of length K
    def has_palindrome_k(s, k):
        # We check every substring of length k
        for i in range(len(s) - k + 1):
            substring = s[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # To avoid iterating through all N! permutations (which is 3.6 million for N=10),
    # we can use a set to get all unique permutations of the string S.
    # Since N is small (up to 10), we can use itertools.permutations and a set.
    
    # Convert string to tuple for permutation processing
    s_tuple = tuple(S)
    
    # Generate all unique permutations
    # We use a set comprehension to filter unique arrangements
    unique_permutations = {''.join(p) for p in permutations(s_tuple)}
    
    # Filter permutations that do not contain a palindrome of length K
    # We use a generator expression inside sum() to count them
    result = sum(1 for p in unique_permutations if not has_palindrome_k(p, K))
    
    # Print the result to stdout
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()