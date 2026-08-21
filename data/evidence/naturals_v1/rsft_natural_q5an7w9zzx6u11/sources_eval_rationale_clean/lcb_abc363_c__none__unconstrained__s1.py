import sys
from itertools import permutations

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set to remove duplicates since S may have repeating characters
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside any() to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition to avoid loops means we use comprehensions and built-ins
    
    # We filter the permutations:
    # For a permutation p, it is valid if for all i from 0 to N-K, 
    # the substring p[i:i+K] is NOT a palindrome.
    
    # We use a list comprehension to iterate through all unique permutations
    # and sum the boolean results (True=1, False=0)
    result = sum([
        1 for p in all_perms 
        if not any(
            p[i:i+k] == p[i:i+k][::-1] 
            for i in range(n - k + 1)
        )
    ])

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()