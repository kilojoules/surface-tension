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
    # Using a set to automatically remove duplicates caused by repeating characters
    all_perms = set(permutations(s))

    # A helper function to check if a string contains a palindrome of length K
    # We use a list comprehension to check every possible substring of length K
    # and all() to ensure none of them are equal to their reverse.
    # The condition "do not contain a palindrome" means for all i, 
    # the substring is NOT equal to its reverse.
    
    # We filter the permutations and count the ones that satisfy the condition.
    # A string T contains a palindrome of length K if any T[i:i+k] == T[i:i+k][::-1]
    # We want the count of strings where NO such i exists.
    
    result = sum(
        1 for p in all_perms 
        if all(
            "".join(p[i:i+k]) != "".join(p[i:i+k])[::-1] 
            for i in range(n - k + 1)
        )
    )

    print(result)

if __name__ == "__main__":
    solve()