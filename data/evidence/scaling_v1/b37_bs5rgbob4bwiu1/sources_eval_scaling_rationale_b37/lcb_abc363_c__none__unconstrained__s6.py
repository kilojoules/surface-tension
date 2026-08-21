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
    # so we use set() to get unique strings.
    all_perms = set(''.join(p) for p in permutations(s))

    # Function to check if a string contains a palindrome of length K
    # A string T contains a palindrome of length K if there is some i
    # such that T[i:i+k] == T[i:i+k][::-1]
    # We use a list comprehension to check all possible substrings of length K
    # and all() to ensure none of them are palindromes.
    
    # To count strings that DO NOT contain a palindrome of length K:
    # We keep permutations where for all i, the substring of length K is not equal to its reverse.
    
    result = [
        p for p in all_perms 
        if all(
            p[i : i + k] != p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # Print the count of valid permutations
    print(len(result))

if __name__ == "__main__":
    solve()