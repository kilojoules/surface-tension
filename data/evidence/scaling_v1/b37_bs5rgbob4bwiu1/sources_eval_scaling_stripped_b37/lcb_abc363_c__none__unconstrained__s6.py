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
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+K] == T[i:i+K][::-1]
    def has_palindrome_k(tup):
        # We check every substring of length K
        for i in range(N - K + 1):
            substring = tup[i:i+K]
            if substring == substring[::-1]:
                return True
        return False

    # Generate all unique permutations of the string S
    # Using a set of tuples to handle duplicate characters in S
    # itertools.permutations treats elements as unique based on position, 
    # so we wrap it in a set to get distinct permutations.
    all_perms = set(permutations(S))
    
    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not has_palindrome_k(p))
    
    # Print the final count to standard output
    print(result)

if __name__ == "__main__":
    solve()