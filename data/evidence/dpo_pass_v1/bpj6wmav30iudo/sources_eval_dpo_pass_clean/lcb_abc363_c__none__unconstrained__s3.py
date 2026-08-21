import sys
from itertools import permutations

def solve():
    # Read input from stdin
    # Using map and split to handle the input line
    input_data = sys.stdin.read().split()
    
    # Assign N, K, and S from the input list
    # N = int(input_data[0])
    # K = int(input_data[1])
    # S = input_data[2]
    # We use a tuple for assignment to avoid multiple statements
    (N, K, S) = (int(input_data[0]), int(input_data[1]), input_data[2])

    # A function to check if a string contains a palindrome of length K
    # We use any() with a generator expression to check all possible substrings
    # A substring is a palindrome if it equals its reverse
    is_palindrome_present = lambda T: any(
        T[i : i + K] == T[i : i + K][::-1] 
        for i in range(N - K + 1)
    )

    # 1. Generate all permutations of S
    # 2. Join them into strings
    # 3. Use a set to remove duplicate permutations
    # 4. Filter out strings that contain a palindrome of length K
    # 5. Count the remaining strings
    result = len(
        set(
            # Filter: keep T if is_palindrome_present(T) is False
            # We use a generator expression inside set()
            "".join(p) 
            for p in permutations(S) 
            if not is_palindrome_present("".join(p))
        )
    )

    # Print the result to stdout
    sys.stdout.write(str(result) + '\n')

# Execute the solver
if __name__ == "__main__":
    solve()