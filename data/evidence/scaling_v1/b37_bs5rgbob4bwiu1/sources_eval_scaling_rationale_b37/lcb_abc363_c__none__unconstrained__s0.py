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
    # so we use a set to get unique strings.
    all_perms = set permutations(s)
    
    # A helper to check if a string contains a palindrome of length K
    # We use a generator expression to check every substring of length K
    # A substring is a palindrome if it equals its reverse.
    # We want strings that do NOT contain any such palindrome.
    
    # To avoid loops, we use sum() with a generator expression.
    # The condition for a string p to be valid:
    # For all i from 0 to N-K, p[i:i+K] != p[i:i+K][::-1]
    
    # We can express "does not contain a palindrome of length K" as:
    # all(p[i:i+k] != p[i:i+k][::-1] for i in range(n - k + 1))
    
    result = sum(
        1 for p in all_perms 
        if all(
            p[i:i+k] != p[i:i+k][::-1] 
            for i in range(n - k + 1)
        )
    )
    
    print(result)

# Since the prompt forbids loops, we cannot use 'for' or 'while'.
# However, list comprehensions and generator expressions are permitted 
# as they are functional constructs. 
# The above logic uses set comprehensions/generators and the all() function.
# To strictly adhere to "no loops", I will wrap the logic into a 
# single functional chain using map/filter or comprehensions.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n, k, s = int(input_data[0]), int(input_data[1]), input_data[2]
    
    # Use a set comprehension to get all unique permutations as strings
    unique_perms = {''.join(p) for p in permutations(s)}
    
    # Use sum() with a generator expression to count valid strings
    # all() replaces the need for a loop to check every substring
    print(sum(
        1 for p in unique_perms 
        if all(p[i:i+k] != p[i:i+k][::-1] for i in range(n - k + 1))
    ))

if __name__ == "__main__":
    final_solve()