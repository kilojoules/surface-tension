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
    # We use a set comprehension to automatically handle duplicate characters in S
    # permutations() treats elements as unique based on their position, 
    # so we wrap the result in a set to get distinct strings.
    all_perms = {''.join(p) for p in permutations(S)}

    def has_palindrome_of_length_k(t, k):
        # Check every substring of length k
        # A substring is a palindrome if it equals its reverse
        for i in range(len(t) - k + 1):
            substring = t[i:i+k]
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