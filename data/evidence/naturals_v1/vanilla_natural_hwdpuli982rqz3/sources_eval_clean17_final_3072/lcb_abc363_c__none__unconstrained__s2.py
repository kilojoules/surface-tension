import sys
from itertools import permutations, combinations_with_replacement
from collections import Counter

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]
    
    # To avoid loops and recursion, we use a list comprehension to generate 
    # all unique permutations. 
    # Since N <= 10, N! is at most 3,628,800, which fits in memory/time.
    
    # We use a set to get unique permutations of the string S.
    # The asterisk unpacks the string S into positional arguments for permutations().
    all_permutations = set(permutations(*S))
    
    # Helper function to check if a tuple (string) contains a palindrome of length K.
    # We use a generator expression inside any() to satisfy the "no loop" constraint.
    # The condition T[i+j] == T[i+K-1-j] checks for palindromes.
    # Note: The problem description uses 1-based indexing for j, 
    # but 0-based indexing in Python is T[i+j] == T[i+K-1-j] for 0 <= j < K // 2.
    
    is_palindrome_free = lambda T: not any(
        all(T[i + j] == T[i + K - 1 - j] for j in range(K // 2))
        for i in range(N - K + 1)
    )
    
    # Filter the unique permutations and count them.
    result = sum(1 for p in all_permutations if is_palindrome_free(p))
    
    # Print the result to stdout.
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()