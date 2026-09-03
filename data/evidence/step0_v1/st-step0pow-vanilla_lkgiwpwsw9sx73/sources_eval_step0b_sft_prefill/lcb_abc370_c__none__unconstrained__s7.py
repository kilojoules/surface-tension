import sys

def solve():
    # Read S and T from standard input
    # Using read().split() to handle input as a list
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # To minimize the number of elements in X, we must change only the characters
    # that differ between S and T.
    # Let D be the set of indices where S[i] != T[i].
    # The number of operations M will be len(D).
    D = [i for i in range(N) if S[i] != T[i]]
    M = len(D)
    
    # To make the array X lexicographically smallest, we want the strings X_j
    # to be as small as possible.
    # A string is smaller if its character at the smallest index is smaller.
    # Therefore, we should prioritize changing characters at smaller indices first
    # IF the target character T[i] is smaller than the current character S[i].
    # If T[i] is larger than S[i], changing it immediately makes the string larger.
    # In that case, we should delay that change as long as possible.
    
    # Correct Strategy for Lexicographical Minimum:
    # 1. First, process all indices i where T[i] < S[i] in increasing order of i.
    # 2. Then, process all indices i where T[i] > S[i] in decreasing order of i.
    
    # Why? 
    # If we change S[i] to T[i] and T[i] < S[i], the string becomes lexicographically smaller.
    # We want this to happen as early as possible, and at the leftmost position possible.
    # If we change S[i] to T[i] and T[i] > S[i], the string becomes lexicographically larger.
    # We want to delay this as much as possible, and if we must do it, at the rightmost position.
    
    part1 = [i for i in D if T[i] < S[i]]
    part2 = [i for i in D if T[i] > S[i]]
    
    # Sort part1 ascending, part2 descending
    order = sorted(part1) + sorted(part2, reverse=True)
    
    # Generate the sequence of strings
    # Since we cannot use loops, we use a list comprehension with a side effect 
    # or a map approach. However, the prompt asks for a complete program.
    # I will use a list comprehension that updates a list and joins it.
    
    s_list = list(S)
    
    # We use a helper function to update and return the string to avoid explicit loops
    def update_and_get(idx):
        s_list[idx] = T[idx]
        return "".join(s_list)
    
    # Map the order to the update function
    result = list(map(update_and_get, order))
    
    # Print results
    print(M)
    sys.stdout.write("\n".join(result) + "\n" if result else "")

if __name__ == "__main__":
    solve()