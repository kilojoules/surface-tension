import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    N = len(S)

    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)

    # To get the lexicographically smallest sequence of strings:
    # 1. We must change characters one by one.
    # 2. For a specific index i, if T[i] < S[i], we want to change it as early as possible
    #    to make the string smaller immediately.
    # 3. If T[i] > S[i], we want to change it as late as possible to avoid making
    #    the string larger sooner than necessary.
    
    # Sort indices: 
    # First, those where T[i] < S[i] (processed in increasing order of index).
    # Then, those where T[i] > S[i] (processed in decreasing order of index).
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing list ascending by index, and increasing list descending by index
    # Wait, the logic for lexicographical smallest array X:
    # To make X_1 smallest, we look at the first index i where we can make S[i] smaller.
    # If we can make S[i] smaller (T[i] < S[i]), we should do it for the smallest i first.
    # If we must make S[i] larger (T[i] > S[i]), we should delay it as much as possible,
    # meaning we process these indices from right to left (largest i first).
    
    order = sorted([i for i in diff_indices if T[i] < S[i]]) + \
            sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)

    # Use a recursive function to generate the sequence of strings
    def generate_sequence(current_s, indices_left):
        if not indices_left:
            return []
        
        idx = indices_left[0]
        # Create new string by replacing character at idx with T[idx]
        new_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return [new_s] + generate_sequence(new_s, indices_left[1:])

    result_x = generate_sequence(S, order)
    
    # Output the number of elements and the elements themselves
    print(len(result_x))
    if result_x:
        print('\n'.join(result_x))

if __name__ == "__main__":
    solve()