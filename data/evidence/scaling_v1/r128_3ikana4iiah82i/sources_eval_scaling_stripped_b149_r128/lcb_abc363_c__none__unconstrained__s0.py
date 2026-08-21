import sys
from itertools import permutations
from functools import reduce

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
    # A string T contains a palindrome of length K if there's an index i such that
    # the substring T[i : i+k] is equal to its reverse.
    def contains_palindrome(t):
        # We check every possible substring of length K
        # The condition T_{i+j} = T_{i+K+1-j} is equivalent to checking if 
        # the substring is a palindrome.
        return any(t[i : i+k] == t[i : i+k][::-1] for i in range(n - k + 1))

    # We need to count permutations that DO NOT contain a palindrome of length K.
    # We use map to apply the check to all permutations and then sum the booleans.
    # Since we want those that do NOT contain it, we use 'not'.
    result = sum(map(lambda p: not contains_palindrome(p), all_perms))

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()