import sys
from itertools import permutations

def solve():
    # Read N and K from the first line, and S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # We use a set to store the permutations as tuples to remove duplicates
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use 'any' with a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition T_{i+j} = T_{i+K+1-j} is equivalent to checking if the substring is a palindrome
    
    # We filter the permutations and count how many do NOT contain a palindrome of length K
    # The check: any(p[i:i+k] == p[i:i+k][::-1] for i in range(n - k + 1))
    # We want the count where this is False.
    
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    )

    # Print the final count
    sys.stdout.write(str(result) + '\n')

if __name__ == "__main__":
    solve()