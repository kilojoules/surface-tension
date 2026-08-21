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
    all_perms = set([''.join(p) for p in permutations(s)])

    # A helper function to check if a string contains a palindrome of length K
    # We use a list comprehension to check every substring of length K
    # and 'any()' to see if any of those substrings are equal to their reverse.
    # However, the constraint is to avoid loops, so we use a generator expression 
    # inside any().
    
    # We filter the set of permutations. 
    # A string is kept if it does NOT contain any palindrome of length K.
    # The condition for a palindrome of length K starting at index i is:
    # T[i : i+k] == T[i : i+k][::-1]
    
    result = [
        p for p in all_perms 
        if not any(
            p[i : i+k] == p[i : i+k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # The answer is the number of strings that satisfied the condition
    print(len(result))

if __name__ == "__main__":
    solve()