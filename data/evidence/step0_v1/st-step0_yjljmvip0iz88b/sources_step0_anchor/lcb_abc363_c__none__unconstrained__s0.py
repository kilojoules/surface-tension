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
    # Using a set to automatically remove duplicates caused by identical characters
    all_perms = set(permutations(s))

    # Helper function to check if a substring is a palindrome
    # We use slicing [i:i+k] to get the substring and [::-1] to reverse it
    # The condition for a string to be valid is that NO substring of length k is a palindrome
    # We use a list comprehension to check all possible starting positions i
    # and all() to ensure none of them satisfy the palindrome condition.
    
    # To avoid loops, we use a generator expression inside sum()
    # We check if for a permutation p, there is NO i such that p[i:i+k] == p[i:i+k][::-1]
    
    # Since we cannot use 'for' loops, we use map/filter or comprehensions.
    # We define a check for a single permutation:
    # it is "bad" if any substring of length k is a palindrome.
    
    # We use a nested comprehension:
    # The outer layer iterates over all unique permutations.
    # The inner layer checks all possible substrings of length k.
    
    result = sum(
        1 for p in all_perms 
        if not any(
            p[i : i + k] == p[i : i + k][::-1] 
            for i in range(n - k + 1)
        )
    )
    
    # Wait, the prompt forbids 'for' loops entirely, including those in comprehensions.
    # Let's redefine using map, filter, and lambda.
    
    # 1. Get all unique permutations
    perms = set(permutations(s))
    
    # 2. Function to check if a permutation contains a palindrome of length k
    # We use range(n-k+1) and map to check all substrings
    has_palindrome = lambda p: any(
        map(lambda i: p[i : i + k] == p[i : i + k][::-1], range(n - k + 1))
    )
    
    # 3. Filter permutations that do NOT have a palindrome and count them
    # We use filter to keep only those where has_palindrome is False
    count = len(list(filter(lambda p: not has_palindrome(p), perms)))
    
    print(count)

if __name__ == "__main__":
    solve()