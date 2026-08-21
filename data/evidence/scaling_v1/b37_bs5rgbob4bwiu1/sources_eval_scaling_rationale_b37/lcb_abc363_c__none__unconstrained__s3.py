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
    # Using a set of tuples to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists an i such that
    # T[i:i+K] == T[i:i+K][::-1]
    # We use a generator expression inside any() to check all possible substrings of length K
    def has_palindrome_k(p):
        return any(
            p[i : i + k] == p[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # Count permutations that do NOT contain a palindrome of length K
    # We use sum() on a generator expression of booleans (True=1, False=0)
    result = sum(
        1 for p in all_perms 
        if not has_palindrome_k(p)
    )

    print(result)

if __name__ == "__main__":
    solve()