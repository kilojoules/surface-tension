import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set of permutations handles duplicate characters in S
    # permutations() returns tuples of characters
    all_perms = set(permutations(S))

    # Function to check if a permutation contains a palindrome of length K
    # We use a generator expression inside any() to check all possible substrings of length K
    # A substring is a palindrome if it reads the same forwards and backwards
    # To avoid loops, we use range() inside a generator expression
    def contains_palindrome(p):
        return any(
            p[i : i + K] == p[i : i + K][::-1]
            for i in range(N - K + 1)
        )

    # Count permutations that do NOT contain a palindrome of length K
    # sum() with a generator expression replaces the for-loop counter
    result = sum(1 for p in all_perms if not contains_palindrome(p))
    
    print(result)

if __name__ == "__main__":
    solve()