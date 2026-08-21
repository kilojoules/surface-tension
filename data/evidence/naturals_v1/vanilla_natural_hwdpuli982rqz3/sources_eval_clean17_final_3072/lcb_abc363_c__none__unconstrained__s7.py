import sys
from itertools import permutations, combinations_with_replacement
from functools import reduce

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
        # s[i : i+k] is the substring
        # It is a palindrome if it equals its reverse
        for i in range(len(s) - k + 1):
            substring = s[i : i+k]
            if substring == substring[::-1]:
                return True
        return False

    # To avoid iterating through all N! permutations (which is 3.6 million for N=10),
    # we can use a set to get all unique permutations of the string S.
    # However, the most straightforward way to get unique permutations in Python
    # without external libraries is using set(permutations(S)).
    
    # Generate all unique permutations of S
    # permutations() returns tuples, we join them into strings
    all_perms = set([''.join(p) for p in permutations(S)])
    
    # Filter permutations that do not contain a palindrome of length K
    valid_perms = [p for p in all_perms if not has_palindrome_k(p, K)]
    
    # Print the count of valid permutations
    print(len(valid_perms))

if __name__ == "__main__":
    solve()