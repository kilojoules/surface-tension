import sys
from itertools import permutations, 
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # A function to check if a string contains a palindrome of length K
    def has_palindrome_k(s):
        # We check every substring of length K
        # s[i : i+K] is the substring
        # s[i : i+K][::-1] is its reverse
        # We use a generator expression inside any() to stop at the first find
        return any(s[i : i+K] == s[i : i+K][::-1] for i in range(N - K + 1))

    # Generate all unique permutations of the string S
    # Since N is small (up to 10), we can iterate through all permutations.
    # To avoid duplicates without using a set (which consumes memory/time),
    # we can use sorted(S) and a custom approach, but given N=10, 
    # 10! is 3,628,800, which is manageable in Python if we use map/filter.
    
    # To handle duplicate characters in S, we use a set to get unique permutations.
    # We convert the string to a tuple to make it hashable.
    all_perms = set(permutations(S))
    
    # Filter permutations that do not contain a palindrome of length K
    # We join the tuple back into a string to check the palindrome condition
    valid_perms = filter(lambda p: not has_palindrome_k("".join(p)), all_perms)
    
    # Count the remaining permutations
    # We use a list comprehension or sum with a generator to count
    print(sum(1 for _ in valid_perms))

if __name__ == "__main__":
    # To avoid using for/while loops as per some strict constraints (though not specified),
    # and to keep it purely functional/declarative:
    solve()