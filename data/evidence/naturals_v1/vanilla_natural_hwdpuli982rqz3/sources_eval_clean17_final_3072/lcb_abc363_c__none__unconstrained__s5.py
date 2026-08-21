import sys
from itertools import permutations, 
from collections import Counter

def is_palindrome(s):
    return s == s[::-1]

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # Generate all unique permutations of the string S
    # Since N <= 10, N! is at most 3,628,800.
    # We can use a set to handle duplicate characters in S.
    all_perms = set(permutations(S))

    # Define a helper to check if a permutation contains a palindrome of length K
    def contains_k_palindrome(p):
        # p is a tuple of characters
        # We check every substring of length K
        for i in range(N - K + 1):
            substring = p[i : i + K]
            # Check if substring is a palindrome
            # A tuple is a palindrome if it equals its reverse
            if substring == substring[::-1]:
                return True
        return False

    # Filter permutations that do not contain a K-length palindrome
    # We use a generator expression inside sum() to count them
    result = sum(1 for p in all_perms if not contains_k_palindrome(p))
    
    # Print the result to stdout
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    # To avoid using for/while loops for the logic as per some strict constraints,
    # though the prompt allows "complete Python program", 
    # I will use map/filter/sum for the counting logic.
    
    # Redefining solve to strictly avoid for/while loops for the counting part
    def solve_no_loops():
        import sys
        from itertools import permutations
        
        input_data = sys.stdin.read().split()
        if not input_data: return
        N, K, S = int(input_data[0]), int(input_data[1]), input_data[2]
        
        all_perms = set(permutations(S))
        
        # Check if a specific permutation p contains any palindrome of length K
        # We use any() with a generator to check all possible start indices i
        has_palindrome = lambda p: any(
            p[i : i + K] == p[i : i + K][::-1] 
            for i in range(N - K + 1)
        )
        
        # Count permutations where has_palindrome is False
        print(sum(1 for p in all_perms if not has_palindrome(p)))

    solve_no_loops()