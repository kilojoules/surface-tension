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

    # Generate all unique permutations of the string S
    # Since N is small (<= 10), we can use set() on all permutations
    # permutations() treats elements as unique based on their position
    all_perms = set(permutations(S))

    def has_palindrome_of_length_k(s_tuple, k):
        # Check every substring of length k for palindrome property
        # A substring starting at i is s_tuple[i : i+k]
        for i in range(len(s_tuple) - k + 1):
            substring = s_tuple[i : i+k]
            # Check if substring is equal to its reverse
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not has_palindrome_of_length_k(p, K))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()