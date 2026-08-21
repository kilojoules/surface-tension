import itertools
import sys

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set comprehension to remove duplicates caused by identical characters
    all_permutations = {''.join(p) for p in itertools.permutations(s)}

    # A helper function to check if a string contains a palindrome of length K
    # We use any() with a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    def contains_palindrome(t):
        return any(
            t[i : i + k] == t[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # Count permutations that do NOT contain a palindrome of length K
    # We use sum() on a generator expression that yields 1 for each valid string
    result = sum(
        1 for t in all_permutations 
        if not contains_palindrome(t)
    )

    print(result)

if __name__ == "__main__":
    solve()