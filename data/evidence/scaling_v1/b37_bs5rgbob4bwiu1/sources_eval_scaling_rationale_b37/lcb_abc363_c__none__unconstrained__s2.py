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
    # permutations() treats elements as unique based on position, 
    # so we use a set to get distinct strings.
    all_perms = set(''.join(p) for p in permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there exists i such that
    # T[i:i+k] == T[i:i+k][::-1]
    # We use any() with a generator expression to check all possible substrings of length K
    is_invalid = lambda t: any(t[i:i+k] == t[i:i+k][::-1] for i in range(n - k + 1))

    # Count permutations that do NOT contain a palindrome of length K
    # We use a list comprehension to filter and sum() to count
    result = sum(1 for p in all_perms if not is_invalid(p))
    
    print(result)

if __name__ == "__main__":
    solve()