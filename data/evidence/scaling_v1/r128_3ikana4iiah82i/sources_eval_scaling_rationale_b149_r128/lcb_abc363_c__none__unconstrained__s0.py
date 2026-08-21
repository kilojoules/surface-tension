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
    # Using a set of tuples to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    
    # We filter the permutations:
    # For each permutation p, we check if ANY substring of length K is a palindrome.
    # We want the count of strings where NO substring of length K is a palindrome.
    
    # The condition "do not contain a palindrome of length K" means:
    # for all i from 0 to N-K: p[i:i+k] != p[i:i+k][::-1]
    
    # We use a list comprehension to create a list of permutations that satisfy the condition
    # and then take the length of that list.
    
    result = [
        p for p in all_perms 
        if not any(
            p[i:i+k] == p[i:i+k][::-1] 
            for i in range(n - k + 1)
        )
    ]
    
    print(len(result))

if __name__ == "__main__":
    solve()