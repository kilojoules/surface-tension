import itertools

def solve():
    # Read N and K from the first line of input
    # Read S from the second line of input
    # Using map and split to handle the input stream
    input_data = __import__('sys').stdin.read().split()
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

    # 1. Generate all permutations of the string S
    # 2. Join the tuples returned by permutations into strings
    # 3. Use a set to keep only unique permutations (since S may have duplicate characters)
    # 4. Filter out strings that contain a palindrome of length K
    # 5. Calculate the length of the resulting list
    result = len([
        p for p in set("".join(perm) for perm in itertools.permutations(S))
        if not is_palindrome_present(p)
    ])

    # Print the final count
    print(result)

# Execute the solver
if __name__ == "__main__":
    solve()