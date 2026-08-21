import itertools
from functools import reduce

def solve():
    # Read N and K from the first line
    # Read S from the second line
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # Function to check if a string contains a palindrome of length K
    def contains_palindrome(s, k):
        # A string contains a palindrome of length k if any substring of length k is a palindrome
        # We check every substring of length k
        for i in range(len(s) - k + 1):
            substring = s[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Generate all unique permutations of the string S
    # itertools.permutations returns all possible orderings based on index.
    # We use a set comprehension to keep only unique strings.
    all_permutations = {''.join(p) for p in itertools.permutations(S)}
    
    # Filter permutations that do not contain a palindrome of length K
    valid_permutations = [p for p in all_permutations if not contains_palindrome(p, K)]
    
    # The answer is the count of such permutations
    print(len(valid_permutations))

if __name__ == "__main__":
    solve()