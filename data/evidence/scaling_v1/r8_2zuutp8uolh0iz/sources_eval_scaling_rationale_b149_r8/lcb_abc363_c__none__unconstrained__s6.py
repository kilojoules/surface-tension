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
    # We use a set to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # A string contains a palindrome of length K if any substring of length K 
    # is equal to its own reverse.
    # We use a list comprehension to filter permutations that do NOT contain such a substring.
    # The inner 'any' check scans all possible starting positions i from 0 to N-K.
    
    # To avoid explicit loops, we use:
    # 1. set(permutations(s)) to get all unique orderings.
    # 2. A list comprehension to iterate through the permutations.
    # 3. 'any()' with a generator expression to check for palindromes of length K.
    # 4. 'sum()' to count the number of valid permutations.

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