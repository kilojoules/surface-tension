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

    # A string T contains a palindrome of length K if there exists i such that
    # T[i + j] == T[i + K - 1 - j] for 0 <= j < K.
    # Note: The problem description uses 1-based indexing T_{i+j} = T_{i+K+1-j}.
    # In 0-based: T[i+j] == T[i + (K-1) - j].
    
    def is_palindrome(sub):
        return sub == sub[::-1]

    def contains_palindrome(tup):
        # Check every substring of length K
        for i in range(N - K + 1):
            substring = tup[i : i + K]
            if substring == substring[::-1]:
                return True
        return False

    # To avoid iterating through all N! permutations (which is 3.6 million for N=10),
    # we can use a set to get all unique permutations.
    # However, itertools.permutations returns all N! tuples.
    # We can convert them to a set to remove duplicates.
    
    # Generate all unique permutations of the string S
    # Since N is small (up to 10), we can afford to generate them.
    all_perms = set(permutations(S))
    
    # Filter permutations that do not contain a palindrome of length K
    valid_perms = [p for p in all_perms if not contains_palindrome(p)]
    
    # Print the count
    print(len(valid_perms))

if __name__ == "__main__":
    solve()