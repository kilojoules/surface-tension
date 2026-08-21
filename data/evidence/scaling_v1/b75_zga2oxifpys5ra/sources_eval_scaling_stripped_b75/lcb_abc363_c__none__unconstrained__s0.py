import sys
from itertools import permutations

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set of tuples because permutations() treats elements as unique by position
    all_perms = set(permutations(s))

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+k] == T[i:i+k][::-1]
    def has_palindrome_k(tup):
        # We check every substring of length k
        for i in range(n - k + 1):
            substring = tup[i:i+k]
            if substring == substring[::-1]:
                return True
        return False

    # Count permutations that do NOT contain a palindrome of length K
    # We use a generator expression inside sum() for efficiency
    result = sum(1 for p in all_perms if not has_palindrome_k(p))
    
    print(result)

if __name__ == "__main__":
    solve()