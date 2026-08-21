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
    # Using a set of permutations handles duplicate characters in S
    all_perms = set(permutations(s))

    def contains_palindrome_of_length_k(t):
        # A string t contains a palindrome of length k if any substring 
        # of length k is equal to its reverse.
        # We check all possible starting positions i from 0 to n-k.
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter the permutations that do NOT contain a palindrome of length k
    # We use a generator expression inside sum() to count the valid permutations
    result = sum(1 for p in all_perms if not contains_palindrome_of_length_k(p))

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()