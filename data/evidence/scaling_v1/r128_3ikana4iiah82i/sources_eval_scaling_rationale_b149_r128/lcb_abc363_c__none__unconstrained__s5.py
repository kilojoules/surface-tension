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
    all_perms = set(''.join(p) for p in permutations(s))

    # A string contains a palindrome of length K if there exists an index i
    # such that the substring of length K starting at i is equal to its reverse.
    # We use a list comprehension to filter perms that do NOT contain any such substring.
    # The condition 'all()' ensures that for a given permutation, 
    # no substring of length K is a palindrome.
    
    result = [
        p for p in all_perms 
        if all(
            p[i:i+k] != p[i:i+k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # Output the count of valid permutations
    print(len(result))

if __name__ == "__main__":
    solve()