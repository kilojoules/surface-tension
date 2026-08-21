import sys
from itertools import permutations

def solve():
    # Read input from standard input
    # Using map and split to avoid explicit loops for input handling
    input_data = sys.stdin.read().split()
    
    # Assign N, K, and S from the input list
    # N = int(input_data[0])
    # K = int(input_data[1])
    # S = input_data[2]
    # We use a tuple for assignment to keep it concise
    N, K, S = int(input_data[0]), int(input_data[1]), input_data[2]

    # A helper function to check if a string contains a palindrome of length K
    # We use 'any' with a generator expression to check all possible substrings of length K
    # A substring is a palindrome if it equals its reverse
    is_palindrome_free = lambda T: not any(
        T[i : i + K] == T[i : i + K][::-1] 
        for i in range(N - K + 1)
    )

    # 1. Generate all permutations of S
    # 2. Join the tuples into strings
    # 3. Use a set to remove duplicate permutations (since S can have repeating characters)
    # 4. Filter the set based on the palindrome-free condition
    # 5. Calculate the length of the resulting filtered collection
    result = len(
        set(
            # Filter strings that do not contain a palindrome of length K
            # We use a generator expression inside set() for efficiency
            "".join(p) 
            for p in permutations(S) 
            if is_palindrome_free("".join(p))
        )
    )

    # Print the final count to standard output
    sys.stdout.write(str(result) + '\n')

# Execute the solver
if __name__ == "__main__":
    solve()