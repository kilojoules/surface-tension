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
    # Since N <= 10, we can use itertools.permutations and then a set to remove duplicates
    # A permutation is a tuple of characters
    all_perms = set(permutations(S))

    def has_palindrome_of_length_k(tup, k):
        # Check every substring of length k
        # A substring starting at i is tup[i : i+k]
        # It is a palindrome if it equals its reverse
        for i in range(len(tup) - k + 1):
            substring = tup[i : i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not has_palindrome_of_length_k(p, K))
    
    print(result)

if __name__ == "__main__":
    solve()