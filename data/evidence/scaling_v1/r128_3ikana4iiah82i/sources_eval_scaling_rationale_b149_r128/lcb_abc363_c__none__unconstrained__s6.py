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
    # Using a set to automatically remove duplicates caused by identical characters
    all_perms = set([''.join(p) for p in permutations(s)])

    # A helper function to check if a string contains a palindrome of length K
    # We use a list comprehension to check every possible substring of length K
    # and 'any' to determine if at least one is a palindrome.
    # A string t is a palindrome if t == t[::-1]
    
    # To follow the constraint of no loops/recursion, we use a 
    # nested comprehension to filter the permutations.
    
    # result is the count of strings where NO substring of length K is a palindrome.
    # We check all i from 0 to N-K.
    
    result = len([
        p for p in all_perms 
        if not any(
            p[i:i+k] == p[i:i+k][::-1] 
            for i in range(n - k + 1)
        )
    ])

    print(result)

if __name__ == "__main__":
    solve()