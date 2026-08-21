import itertools
import sys

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # itertools.permutations returns tuples of characters
    # Using a set comprehension to remove duplicates caused by identical characters in S
    all_permutations = {''.join(p) for p in itertools.permutations(s)}

    # A string T contains a palindrome of length K if there exists an index i
    # such that the substring T[i : i+K] is equal to its reverse.
    # We use a list comprehension to filter the permutations.
    # The condition 'not any(...)' ensures no substring of length K is a palindrome.
    result = [
        t for t in all_permutations
        if not any(
            t[i : i + k] == t[i : i + k][::-1]
            for i in range(n - k + 1)
        )
    ]

    # Output the count of valid permutations
    print(len(result))

if __name__ == "__main__":
    solve()