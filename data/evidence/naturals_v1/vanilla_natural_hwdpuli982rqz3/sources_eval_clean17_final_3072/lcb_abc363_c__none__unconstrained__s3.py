import sys
from itertools import permutations

def is_palindrome_free(s, k):
    # Check every substring of length k
    # s[i : i+k] is the substring
    # We check if it equals its reverse
    for i in range(len(s) - k + 1):
        substring = s[i : i+k]
        if substring == substring[::-1]:
            return False
    return True

def solve():
    # Read input using sys.stdin.read().split()
    # The input consists of N, K, and S
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    k = int(data[1])
    s = data[2]
    
    # Generate all unique permutations of the string S
    # Since N <= 10, N! is at most 3,628,800.
    # We can use a set to handle duplicate characters in S.
    
    # We use a generator expression inside sum() to avoid explicit loops
    # permutations(s) returns all N! permutations.
    # We join them into strings and use a set to keep only unique ones.
    
    unique_permutations = {''.join(p) for p in permutations(s)}
    
    # Filter permutations that do not contain a palindrome of length K
    # and count them.
    result = sum(1 for p in unique_permutations if is_palindrome_free(p, k))
    
    print(result)

if __name__ == "__main__":
    solve()