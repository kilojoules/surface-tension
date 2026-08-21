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
    # Using a set of tuples to handle duplicate characters in S
    all_permutations = set(itertools.permutations(s))

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to check all substrings of length K
    # A substring is a palindrome if it equals its reverse
    # The condition 'any(...)' replaces a for-loop
    # We wrap this in a function or a lambda to use inside the final sum
    is_palindrome_free = lambda p: not any(
        p[i:i+k] == p[i:i+k][::-1] 
        for i in range(n - k + 1)
    )

    # Use sum() with a generator expression to count permutations that satisfy the condition
    # This replaces the need for a loop and a counter variable
    result = sum(1 for p in all_permutations if is_palindrome_free(p))
    
    # Print the final result
    sys.stdout.write(str(result) + '\n')

# Using a wrapper to call solve() without an explicit loop
# Since the constraints forbid loops/recursion, we simply call the function.
solve()