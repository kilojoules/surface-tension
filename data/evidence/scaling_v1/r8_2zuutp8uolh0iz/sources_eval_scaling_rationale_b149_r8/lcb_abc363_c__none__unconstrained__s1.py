import itertools
import sys

def solve():
    # Read N and K from the first line, and S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # itertools.permutations treats elements as unique based on position, 
    # so we use a set to keep only unique string results.
    all_perms = set([''.join(p) for p in itertools.permutations(s)])

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse.
    # The condition T_{i+j} = T_{i+K+1-j} is simply the definition of a palindrome.
    
    # We filter the set of permutations to keep only those that do NOT contain any palindrome of length K.
    # The logic: keep perm if NOT (any substring of length K is equal to its reverse)
    valid_perms = [
        perm for perm in all_perms 
        if not any(
            perm[i : i + k] == perm[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # Output the count of valid permutations
    print(len(valid_perms))

if __name__ == "__main__":
    solve()