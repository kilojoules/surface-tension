import sys
from itertools import permutations

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # A string contains a palindrome of length K if any substring of length K is equal to its reverse.
    # We use a set of permutations to handle duplicate characters in S.
    # We use a list comprehension to filter permutations that do not contain any K-length palindrome.
    # 'all()' is used to ensure no substring of length K is a palindrome.
    # 'any()' is used inside 'all()' to check if any starting position i yields a palindrome.
    
    # To avoid loops, we use map, filter, and list comprehensions.
    # We generate all unique permutations of S.
    unique_perms = set(permutations(S))
    
    # For each permutation P, we check if it contains a palindrome of length K.
    # P is a tuple of characters. P[i:i+K] is the substring.
    # The condition for a palindrome is P[i:i+K] == P[i:i+K][::-1].
    
    # We want the count of permutations where for all i from 0 to N-K, 
    # the substring of length K is NOT a palindrome.
    result = len([
        p for p in unique_perms 
        if all(
            p[i : i + K] != p[i : i + K][::-1] 
            for i in range(N - K + 1)
        )
    ])
    
    # Print the result to standard output
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    # Execute the solve function
    # Since the prompt forbids for/while loops and recursion, 
    # the logic is contained within comprehensions and built-in functionals.
    solve()