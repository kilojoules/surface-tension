import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # Function to check if a string contains a palindrome of length K
    def has_palindrome_k(s, k):
        # A string contains a palindrome of length k if any substring of length k is a palindrome
        # We check every substring of length k
        for i in range(len(s) - k + 1):
            substring = s[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Generate all unique permutations of the string S
    # Using a set comprehension to handle duplicate characters in S
    # permutations() treats elements as unique based on their position, 
    # so we join them into strings and put them in a set.
    all_permutations = {''.join(p) for p in permutations(S)}

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_permutations if not has_palindrome_k(p, K))

    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()