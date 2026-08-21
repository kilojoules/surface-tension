import sys
from itertools import permutations

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set to ensure we only count distinct strings
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside any() to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition to avoid loops means we use comprehensions and built-in functions
    
    # We filter the permutations:
    # For each permutation p, we check if ANY substring of length K is a palindrome.
    # We want the count of permutations where NO substring of length K is a palindrome.
    
    # join the tuple p into a string
    # check all i from 0 to N-K
    # substring is p[i : i+k]
    
    result = sum(
        1 for p in all_perms 
        if not any(
            "".join(p[i : i + k]) == "".join(p[i : i + k])[::-1] 
            for i in range(n - k + 1)
        )
    )

    print(result)

if __name__ == "__main__":
    solve()