import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read N and K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set of tuples to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there's an index i
    # such that the substring T[i : i+k] is equal to its reverse.
    def contains_palindrome(p):
        # We check every possible substring of length K
        # The condition T_{i+j} = T_{i+K+1-j} is equivalent to 
        # the substring being a palindrome.
        for i in range(n - k + 1):
            substring = p[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # We want to count permutations that DO NOT contain a palindrome of length K.
    # We use a generator expression inside sum() to count the valid permutations.
    # The logic is: for each permutation p in all_perms, 
    # if contains_palindrome(p) is False, it contributes 1 to the sum.
    result = sum(1 for p in all_perms if not contains_palindrome(p))

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()