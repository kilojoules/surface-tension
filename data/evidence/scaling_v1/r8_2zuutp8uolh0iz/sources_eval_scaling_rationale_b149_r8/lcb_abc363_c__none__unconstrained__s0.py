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
    # We use a set to automatically handle duplicate characters in S
    all_perms = set("".join(p) for p in permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use any() with a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition "do not contain" means we want the negation of the palindrome check
    
    # We count permutations that satisfy the condition:
    # For all i from 0 to N-K, the substring starting at i of length K is NOT a palindrome.
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    )

    print(result)

if __name__ == "__main__":
    solve()