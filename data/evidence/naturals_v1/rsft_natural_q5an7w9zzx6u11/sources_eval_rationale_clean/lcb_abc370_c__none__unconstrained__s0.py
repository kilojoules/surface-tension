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
    # We need to change characters at these indices to match T
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To get the lexicographically smallest array X, we must consider:
    # 1. If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #    We should do these changes as early as possible.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should do these changes as late as possible.
    
    # Sort indices: 
    # First, indices where T[i] < S[i] (processed in increasing order of index)
    # Then, indices where T[i] > S[i] (processed in decreasing order of index)
    # Wait, the rule for lexicographical smallest array is:
    # At each step, we want the resulting string to be as small as possible.
    # If we can make a character smaller, we should do it immediately (from left to right).
    # If we must make a character larger, we should delay it as long as possible (from right to left).
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (left to right) to make string smaller ASAP
    # Sort increasing indices descending (right to left) to keep string smaller longer
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Use reduce to generate the sequence of strings.
    # The accumulator stores the list of strings generated so far.
    def evolve(acc, idx):
        current_s = acc[-1] if acc else S
        # Create new string by replacing character at idx
        next_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [next_s]

    result_array = reduce(evolve, order, [])
    
    # Output the number of elements and the elements themselves
    print(len(result_array))
    # Use join and map to avoid loops for printing
    if result_array:
        sys.stdout.write('\n'.join(result_array) + '\n')

if __name__ == "__main__":
    solve()