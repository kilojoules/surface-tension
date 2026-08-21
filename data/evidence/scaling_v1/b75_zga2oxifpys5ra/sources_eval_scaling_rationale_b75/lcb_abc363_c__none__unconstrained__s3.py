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
    # We use a set of tuples because permutations() treats elements as unique by position
    all_perms = set(permutations(s))

    # Function to check if a string (tuple) contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+k] == T[i:i+k][::-1]
    # We use a generator expression inside 'any' to check all possible starting positions i
    is_palindrome_free = lambda t: not any(
        t[i : i + k] == t[i : i + k][::-1]
        for i in range(n - k + 1)
    )

    # Filter the permutations and count the ones that are palindrome-free
    # We use a list comprehension (or generator) inside len() via a set/list
    # Since we need the count, we can sum 1s for every permutation that satisfies the condition
    result = sum(1 for p in all_perms if is_palindrome_free(p))
    
    print(result)

if __name__ == "__main__":
    solve()