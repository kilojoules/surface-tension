import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]

    # To get the lexicographically smallest array X:
    # 1. If T[i] < S[i], changing S[i] to T[i] makes the string smaller.
    #    We want to do this as early as possible.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    #    We want to do this as late as possible.
    # Therefore, we sort indices: those with T[i] < S[i] first (in increasing order of i),
    # then those with T[i] > S[i] last (in decreasing order of i).
    
    # Sort criteria: 
    # Primary: T[i] < S[i] comes before T[i] > S[i]
    # Secondary: If T[i] < S[i], smaller i first. If T[i] > S[i], larger i first.
    sorted_indices = sorted(
        diff_indices, 
        key=lambda i: (0, i) if T[i] < S[i] else (1, -i)
    )

    # Use reduce to generate the sequence of strings.
    # The accumulator stores the list of strings generated so far.
    # The current string is tracked by looking at the last element of the list.
    def generate_sequence(acc, idx):
        current_s = acc[-1] if acc else S
        # Create a new string by changing character at idx to T[idx]
        next_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [next_s]

    result_array = reduce(generate_sequence, sorted_indices, [])

    # Output the number of elements and the elements themselves
    print(len(result_array))
    if result_array:
        print('\n'.join(result_array))

if __name__ == "__main__":
    solve()