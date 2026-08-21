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
    # We use a set to ensure we only count distinct strings
    all_permutations = {''.join(p) for p in permutations(s)}

    # Function to check if a string contains a palindrome of length K
    # We use any() with a generator expression to check all substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The range of start indices is from 0 to N-K
    contains_palindrome = lambda t: any(
        t[i:i+k] == t[i:i+k][::-1] 
        for i in range(n - k + 1)
    )

    # Count permutations that do NOT contain a palindrome of length K
    # We use sum() on a generator expression that yields 1 for valid strings
    result = sum(1 for t in all_permutations if not contains_palindrome(t))
    
    print(result)

if __name__ == "__main__":
    solve()