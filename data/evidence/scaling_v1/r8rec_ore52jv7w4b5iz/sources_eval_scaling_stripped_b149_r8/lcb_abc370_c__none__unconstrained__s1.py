import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest:
    # 1. We must change characters at indices that make the resulting string smaller.
    # 2. If S[i] > T[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #    These should be handled as early as possible (from left to right).
    # 3. If S[i] < T[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    These should be handled as late as possible (from right to left).
    
    # Group 1: Indices where S[i] > T[i] (processed left to right)
    # Group 2: Indices where S[i] < T[i] (processed right to left)
    group1 = [i for i in diff_indices if S[i] > T[i]]
    group2 = [i for i in diff_indices if S[i] < T[i]]
    
    # The order of indices to change
    order = group1 + group2[::-1]
    
    # Generate the sequence of strings
    # We use a list comprehension to create the strings by updating S character by character
    # Since we cannot use loops, we simulate the state by tracking how many indices from 'order' have been applied
    
    # For each step k (1 to len(order)), the string is S with characters at order[0...k-1] replaced by T
    # We create a helper function to build the string for a given set of modified indices
    def get_string(k):
        modified_indices = set(order[:k])
        return "".join([T[i] if i in modified_indices else S[i] for i in range(len(S))])

    # Generate the array X
    X = [get_string(k) for k in range(1, len(order) + 1)]
    
    # Output the number of elements and the elements themselves
    print(len(X))
    if X:
        print("\n".join(X))

if __name__ == "__main__":
    solve()