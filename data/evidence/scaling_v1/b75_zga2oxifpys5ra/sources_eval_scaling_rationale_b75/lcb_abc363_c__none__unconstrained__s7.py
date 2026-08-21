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
    # We use a set to ensure we only count distinct strings
    all_perms = set(permutations(s))

    # A function to check if a string (tuple of chars) contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    is_palindrome_free = lambda p: not any(
        p[i:i+k] == p[i:i+k][::-1] 
        for i in range(n - k + 1)
    )

    # Filter the permutations and count the ones that are palindrome-free
    # We use a generator expression inside sum() or len() with a list comprehension
    # Since we cannot use loops, we use a list comprehension to filter and len() to count
    result = len([p for p in all_perms if is_palindrome_free(p)])
    
    print(result)

if __name__ == "__main__":
    solve()