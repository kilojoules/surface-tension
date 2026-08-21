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
    all_perms = set(permutations(s))

    # Function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # We use join() to convert the permutation tuple back into a string
    
    # The condition to filter: 
    # Keep the permutation if for all i from 0 to N-K, 
    # the substring of length K starting at i is NOT a palindrome.
    
    # We use a list comprehension to count how many permutations satisfy the condition.
    # The condition 'not any(...)' checks that no substring of length K is a palindrome.
    
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