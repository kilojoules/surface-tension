import sys
from itertools import permutations, 
from collections import Counter

def is_palindrome_substring_exists(s, k):
    # Check every substring of length k to see if it is a palindrome
    for i in range(len(s) - k + 1):
        substring = s[i : i + k]
        if substring == substring[::-1]:
            return True
    return False

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]
    
    # Generate all unique permutations of the string S
    # Since N <= 10, N! is at most 3,628,800, which is manageable.
    # We use a set to handle duplicate characters in S.
    all_perms = set(permutations(s))
    
    # Filter permutations that do not contain a palindrome of length K
    # We join the tuple from permutations into a string and check the condition
    valid_perms = [
        p for p in all_perms 
        if not is_palindrome_substring_exists("".join(p), k)
    ]
    
    # Print the count of valid permutations
    print(len(valid_perms))

if __name__ == "__main__":
    solve()