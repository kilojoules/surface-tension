import itertools
import sys

def solve():
    # Read N and K from the first line
    # Read S from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # Helper function to check if a string contains a palindrome of length K
    # We use a generator expression inside 'any' to avoid explicit loops
    def contains_palindrome(t, k):
        return any(
            t[i : i + k] == t[i : i + k][::-1]
            for i in range(n - k + 1)
        )

    # 1. Generate all unique permutations of the string S
    # 2. Filter out those that contain a palindrome of length K
    # 3. Count the remaining strings
    # We use a set comprehension to get unique permutations and 
    # a generator expression within sum() to count valid ones.
    
    all_permutations = set(itertools.permutations(s))
    
    # Convert tuples back to strings and check the palindrome condition
    result = sum(
        1 for p in all_permutations 
        if not contains_palindrome("".join(p), k)
    )
    
    print(result)

if __name__ == "__main__":
    solve()