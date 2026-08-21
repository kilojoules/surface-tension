import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read N and K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set of tuples to handle duplicate characters in S
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i : i+K] == T[i : i+K][::-1]
    # However, the problem defines the palindrome condition as:
    # T_{i+j} = T_{i+K+1-j} for 1 <= j <= K.
    # Let's map this to 0-indexing:
    # For a substring starting at index i (0 <= i <= N-K):
    # The characters are T[i], T[i+1], ..., T[i+K-1]
    # The condition T_{i+j} = T_{i+K+1-j} for j=1..K means:
    # j=1: T[i] == T[i+K-1]
    # j=2: T[i+1] == T[i+K-2]
    # ... which is exactly the definition of a palindrome.
    
    def has_palindrome_k(p):
        # p is a tuple of characters
        # We check every substring of length K
        for i in range(n - k + 1):
            substring = p[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Filter permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() to count valid permutations
    result = sum(1 for p in all_perms if not has_palindrome_k(p))
    
    print(result)

if __name__ == "__main__":
    solve()