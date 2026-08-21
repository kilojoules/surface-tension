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
    # Using a set comprehension to remove duplicates
    all_perms = {''.join(p) for p in permutations(s)}

    # A helper function to check if a string contains a palindrome of length K
    # We use 'any' inside a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition to avoid loops forces the use of sum() and generator expressions
    
    # We count permutations where NO substring of length K is a palindrome
    # The logic: sum(1 for p in all_perms if not any(p[i:i+k] == p[i:i+k][::-1] for i in range(n-k+1)))
    
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