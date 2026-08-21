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
    # Using a set to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A string is a palindrome if it equals its reverse
    # We check if any substring of length K is a palindrome
    # The condition is: there exists i such that T[i:i+K] == T[i:i+K][::-1]
    
    # We count permutations that do NOT contain any palindrome of length K
    # sum() treats True as 1 and False as 0
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    )

    print(result)

if __name__ == "__main__":
    solve()