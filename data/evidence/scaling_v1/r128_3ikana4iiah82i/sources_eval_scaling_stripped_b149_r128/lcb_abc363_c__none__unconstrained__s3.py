import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read N and K
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there is an index i
    # such that T[i : i+k] == T[i : i+k][::-1]
    def contains_palindrome(t):
        # We check all possible substrings of length K
        # The condition T_{i+j} = T_{i+K+1-j} is the definition of a palindrome
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if substring == substring[::-1]:
                return True
        return False

    # Generate all unique permutations of S
    # We use a set of tuples to handle duplicate characters in S
    all_perms = set(permutations(s))
    
    # We need to count permutations that DO NOT contain a palindrome of length K
    # We use map to apply the check and then filter for those that are False
    # Finally, we count the remaining items.
    # Using a generator expression inside sum() is an efficient way to count.
    result = sum(1 for p in all_perms if not contains_palindrome("".join(p)))
    
    print(result)

if __name__ == "__main__":
    solve()