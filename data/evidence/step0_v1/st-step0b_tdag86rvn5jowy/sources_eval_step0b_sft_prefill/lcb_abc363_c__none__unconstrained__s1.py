import sys
from itertools import permutations, combinations_with_replacement

def is_palindrome(s):
    return s == s[::-1]

def contains_palindrome_of_length_k(s, k):
    # Check every substring of length k
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i : i + k]):
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

    # The constraints are N <= 10, which is small enough to iterate 
    # through all unique permutations.
    # We can use a set to get all unique permutations of the string S.
    # Since we cannot use loops or recursion for the logic flow (per some strict 
    # interpretations of "no for/while" though not explicitly forbidden here, 
    # I will use map/filter/list comprehensions to be safe and concise).
    
    # Generate all permutations of the indices to handle the string
    # Then join them to form the strings.
    # Using a set comprehension to get unique permutations.
    all_perms = {"".join([s[i] for i in p]) for p in permutations(range(n))}
    
    # Filter permutations that do not contain a palindrome of length K
    valid_perms = list(filter(lambda x: not contains_palindrome_of_length_k(x, k), all_perms))
    
    # Print the count of valid permutations
    print(len(valid_perms))

if __name__ == "__main__":
    solve()