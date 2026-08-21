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

    # To avoid using loops/recursion for permutations, we use itertools.permutations
    # Since N <= 10, N! is at most 3,628,800, which fits in memory/time.
    
    # We generate all unique permutations of the string S.
    # To do this without loops, we can use a set comprehension.
    # Note: permutations() returns tuples.
    
    all_perms = {''.join(p) for p in permutations(S)}
    
    # Filter permutations that do not contain a palindrome of length K
    valid_perms = [p for p in all_perms if not has_palindrome_k(p, K)]
    
    # Print the count of valid permutations
    sys.stdout.write(str(len(valid_perms)) + '\n')

if __name__ == "__main__":
    solve()