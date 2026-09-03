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
    # Since N is small (up to 10), we can use itertools.permutations 
    # and then use a set to filter unique strings.
    # 10! is 3,628,800, which is manageable in Python within a few seconds.
    
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # p is a tuple of characters
        # Check if any substring of length k is a palindrome
        has_palindrome = False
        for i in range(n - k + 1):
            # Extract substring of length k
            substring = p[i : i + k]
            # Check if it's a palindrome
            # A tuple is a palindrome if it equals its reverse
            if substring == substring[::-1]:
                has_palindrome = True
                break
        
        if not has_palindrome:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()