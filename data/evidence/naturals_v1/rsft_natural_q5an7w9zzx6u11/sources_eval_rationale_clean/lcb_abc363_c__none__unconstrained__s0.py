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
    # Using a set to ensure we only count distinct strings
    all_perms = set(''.join(p) for p in permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a list comprehension to check every possible substring of length K
    # and all() to ensure none of them are equal to their reverse.
    # The condition is: for a string T, it is valid if for all i, 
    # T[i:i+k] != T[i:i+k][::-1]
    
    # We filter the set of permutations and count the survivors
    result = len([
        t for t in all_perms 
        if all(t[i:i+k] != t[i:i+k][::-1] for i in range(n - k + 1))
    ])

    print(result)

if __name__ == "__main__":
    solve()