import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, k = map(int, line1)
        # Read the string S
        s = sys.stdin.readline().strip()
    except ValueError:
        return

    # To avoid explicit loops and recursion, we use functional programming constructs.
    # 1. Generate all unique permutations of the string S.
    # 2. For each permutation, check if it contains a palindrome of length K.
    # 3. Count how many permutations do NOT contain such a palindrome.

    # Using a set of permutations handles duplicate characters in S.
    # We use a generator expression inside sum() to count valid strings.
    
    # A helper function to check if a substring is a palindrome.
    # Since we can't use 'def' for the logic inside the count, 
    # we can use a lambda or a slice comparison.
    
    # The condition for a palindrome of length K starting at index i is:
    # T[i : i+k] == T[i : i+k][::-1]
    
    # We use set(permutations(s)) to get all distinct permutations.
    # We use any() to check if any substring of length K is a palindrome.
    # We use not any(...) to filter for strings that do NOT contain a palindrome.
    
    all_perms = set(permutations(s))
    
    # We use a generator expression to iterate through all unique permutations.
    # For each permutation p (which is a tuple of characters), 
    # we check all possible starting positions i from 0 to N-K.
    
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i+k] == p[i : i+k][::-1] 
            for i in range(n - k + 1)
        )
    )
    
    print(result)

if __name__ == "__main__":
    solve()