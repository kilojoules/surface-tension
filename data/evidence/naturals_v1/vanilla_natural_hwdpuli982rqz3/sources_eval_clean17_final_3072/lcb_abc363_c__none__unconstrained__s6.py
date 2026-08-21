import sys
from itertools import permutations

def is_palindrome_substring_present(s, k):
    """
    Checks if the string s contains a palindrome of length k as a substring.
    """
    # We check every substring of length k
    # s[i : i+k] is the substring
    # s[i : i+k][::-1] is the reversed substring
    return any(s[i : i+k] == s[i : i+k][::-1] for i in range(len(s) - k + 1))

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]
    
    # Generate all unique permutations of the string S
    # Since N <= 10, N! is at most 3,628,800, which is feasible.
    # We use a set to handle duplicate characters in S.
    all_perms = set(permutations(s))
    
    # Convert each permutation tuple back to a string and filter those 
    # that do not contain a palindrome of length K.
    # We use a generator expression inside sum() to count them.
    result = sum(1 for p in all_perms if not is_palindrome_substring_present("".join(p), k))
    
    # Print the result to stdout
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()