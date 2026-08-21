import sys
from itertools import permutations

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    # A function to check if a string contains a palindrome of length K
    # We use any() with a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    is_palindrome_present = lambda T: any(
        T[i : i + K] == T[i : i + K][::-1] 
        for i in range(N - K + 1)
    )

    # 1. Generate all permutations of S
    # 2. Join them into strings
    # 3. Use a set to remove duplicate permutations (since S may have repeating characters)
    # 4. Filter out strings that contain a palindrome of length K
    # 5. Count the remaining strings
    result = len(
        [
            s for s in set("".join(p) for p in permutations(S))
            if not is_palindrome_present(s)
        ]
    )

    # Print the final count to standard output
    sys.stdout.write(str(result) + '\n')

# Execute the solver
if __name__ == "__main__":
    solve()