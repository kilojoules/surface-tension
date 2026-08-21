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
    # itertools.permutations treats elements as unique based on position, 
    # so we use a set to get distinct permutations of the characters.
    all_perms = set(permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside any() to check all substrings of length K
    # A substring is a palindrome if it equals its reverse.
    def has_palindrome_k(p_tuple):
        # Join tuple to string for easier slicing
        t = "".join(p_tuple)
        return any(
            t[i : i + k] == t[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # Count permutations that do NOT contain a palindrome of length K
    # sum() on a generator of booleans effectively counts the True values
    result = sum(1 for p in all_perms if not has_palindrome_k(p))
    
    print(result)

if __name__ == "__main__":
    solve()