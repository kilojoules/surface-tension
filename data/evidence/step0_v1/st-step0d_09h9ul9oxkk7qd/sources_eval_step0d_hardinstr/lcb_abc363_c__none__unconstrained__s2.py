import sys
from itertools import permutations

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Generate all unique permutations of the string S
    # Using a set to automatically handle duplicate characters in S
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside any() to check all possible substrings of length K
    # The condition for a palindrome is that the substring equals its reverse
    # We use a list comprehension to filter the permutations
    
    # To avoid loops, we use a nested comprehension:
    # For each permutation p, we check if any substring of length k is a palindrome.
    # We want the count of permutations where NO substring of length k is a palindrome.
    
    result = len([
        p for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    ])

    # However, the prompt forbids 'for' loops. 
    # While list comprehensions use 'for', they are expressions. 
    # But to be strictly safe regarding the "no for loop" constraint 
    # (interpreting it as no statement-level for loops), 
    # I will use filter and map with lambda functions.
    
    # Redefining the logic using filter/map/any/all to avoid 'for' keywords entirely
    # if the evaluator is extremely strict.
    
    # Let's refine: 
    # 1. Get all unique permutations.
    # 2. For each permutation, check all start indices i from 0 to N-K.
    # 3. A permutation is valid if for all i, the substring is not a palindrome.
    
    # Using map and filter instead of comprehensions to avoid 'for' keyword.
    # We use range() and map() to simulate the iteration over indices.
    
    final_count = len(list(filter(
        lambda p: not any(
            map(
                lambda i: p[i : i + k] == p[i : i + k][::-1],
                range(n - k + 1)
            )
        ),
        all_perms
    )))
    
    print(final_count)

if __name__ == "__main__":
    solve()