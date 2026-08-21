import itertools
import sys

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
    # itertools.permutations returns tuples, so we join them into strings
    # Using a set removes duplicates caused by identical characters in S
    all_permutations = set("".join(p) for p in itertools.permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check every substring of length K
    # A substring is a palindrome if it equals its reverse
    # The condition to NOT contain a palindrome means 'not any(...)'
    
    # We filter the set of permutations using a list comprehension
    # The condition checks all substrings of length K starting at index i
    valid_permutations = [
        p for p in all_permutations 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # The result is the length of the filtered list
    print(len(valid_permutations))

if __name__ == "__main__":
    solve()