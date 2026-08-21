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
    # Using a set of tuples to automatically remove duplicates
    all_perms = set(permutations(s))

    # A string T contains a palindrome of length K if there exists an index i
    # such that T[i:i+k] is equal to its reverse.
    # We want to count permutations that do NOT satisfy this for any i.
    
    # We use a list comprehension to filter the permutations.
    # For each permutation p, we check if any substring of length k is a palindrome.
    # p[i:i+k] == p[i:i+k][::-1] checks if the substring is a palindrome.
    # any(...) returns True if at least one palindrome of length k is found.
    # We negate this with 'not' to keep only those without such palindromes.
    
    result = [
        p for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    ]

    # The answer is the length of the filtered list
    print(len(result))

if __name__ == "__main__":
    solve()