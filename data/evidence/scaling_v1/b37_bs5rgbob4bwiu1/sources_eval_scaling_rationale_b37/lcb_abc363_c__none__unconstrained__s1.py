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

    # Generate all unique permutations of S
    # Using a set to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+k] == T[i:i+k][::-1]
    
    # We use a list comprehension to check all possible substrings of length K
    # and the any() function to determine if any of them are palindromes.
    # To avoid loops, we use map() or list comprehensions.
    
    # The condition to filter: 
    # NOT (any substring of length K is a palindrome)
    
    # We can express the "contains palindrome" check as:
    # any([''.join(p[i:i+k]) == ''.join(p[i:i+k][::-1]) for i in range(n - k + 1)])
    
    # To count valid permutations:
    # We iterate through the set of permutations and count those that satisfy the condition.
    
    # Since we cannot use loops, we use a list comprehension to create a list of booleans
    # and sum() to count the True values.
    
    result = sum([
        1 for p in all_perms 
        if not any([
            p[i + j] == p[i + k - 1 - j] 
            for i in range(n - k + 1) 
            for j in range(k // 2)
        ])
    ])
    
    # Wait, the logic above for 'any' needs to be carefully structured to check 
    # if ANY substring is a palindrome. The nested comprehension above checks 
    # if ANY character pair in ANY substring matches. That's wrong.
    # It should be: for a specific i, ALL j must match. If that happens for ANY i, it's a palindrome.
    
    # Correct logic using functional style:
    # A permutation p is invalid if there exists i such that for all j, p[i+j] == p[i+k-1-j].
    # We want the count of p where for all i, there exists j such that p[i+j] != p[i+k-1-j].
    
    # Let's redefine the filter:
    # Valid if: for all i in 0..N-K: substring p[i:i+K] is NOT a palindrome.
    # Substring is palindrome if: all(p[i+j] == p[i+k-1-j] for j in range(k//2))
    
    # Final functional approach:
    ans = sum([
        1 for p in all_perms
        if not any([
            all([p[i + j] == p[i + k - 1 - j] for j in range(k // 2)])
            for i in range(n - k + 1)
        ])
    ])
    
    print(ans)

if __name__ == "__main__":
    solve()