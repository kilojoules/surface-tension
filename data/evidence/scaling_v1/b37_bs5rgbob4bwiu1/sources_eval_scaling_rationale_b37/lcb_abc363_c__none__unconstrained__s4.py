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
    # We use a generator expression inside any() to check all substrings of length K
    def has_palindrome(t):
        return any(
            t[i : i + K] == t[i : i + K][::-1]
            for i in range(N - K + 1)
        )

    # itertools.permutations treats elements as unique based on position, not value.
    # To handle duplicate characters in S, we use a set of the resulting tuples.
    # We then use a generator expression to count how many of these unique 
    # permutations do not contain a palindrome of length K.
    
    # 1. Generate all unique permutations of S
    # 2. Filter those that do not have a palindrome of length K
    # 3. Sum the boolean results (True=1, False=0)
    
    all_perms = set(permutations(S))
    
    # We join the tuple of characters into a string to use slicing for palindrome check
    result = sum(
        1 for p in all_perms 
        if not has_palindrome("".join(p))
    )
    
    print(result)

if __name__ == "__main__":
    solve()