import sys
from itertools import permutations

def solve():
    # Read N and K from the first line, and S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # A string contains a palindrome of length K if any substring of length K 
    # is equal to its own reverse.
    # We use a list comprehension to filter permutations that DO NOT contain such a palindrome.
    # The inner 'any' checks all possible substrings of length K.
    
    # To avoid loops, we use:
    # 1. set(permutations(s)) to get all unique arrangements.
    # 2. A list comprehension to iterate through the permutations.
    # 3. 'any()' with a generator expression to check for palindromes.
    # 4. 'sum()' or 'len()' to count the valid ones.

    result = len([
        p for p in all_perms 
        if not any(
            "".join(p[i:i+k]) == "".join(p[i:i+k])[::-1] 
            for i in range(n - k + 1)
        )
    ])

    print(result)

if __name__ == "__main__":
    solve()