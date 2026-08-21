import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set to remove duplicates since S may contain repeated characters
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # Note: p is a tuple of characters from permutations()
    
    # We filter the permutations: 
    # Keep permutation p if for all i from 0 to N-K, the substring p[i:i+k] is NOT a palindrome.
    # This is equivalent to: NOT (any substring of length K is a palindrome)
    
    valid_perms = [
        p for p in all_perms 
        if not any(
            p[i + j] == p[i + k - 1 - j] 
            for i in range(n - k + 1) 
            for j in range(k // 2)
        )
    ]

    # The answer is the length of the list of valid permutations
    print(len(valid_perms))

if __name__ == "__main__":
    solve()