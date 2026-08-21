import sys
from functools import reduce

def solve():
    # Read input strings
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    # We need to change these indices one by one.
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest, we must consider:
    # 1. If T[i] < S[i], changing index i makes the string lexicographically smaller.
    #    We should do these as early as possible.
    # 2. If T[i] > S[i], changing index i makes the string lexicographically larger.
    #    We should do these as late as possible.
    # 3. Among those that make it smaller, we should process them from left to right
    #    to affect the most significant character first.
    # 4. Among those that make it larger, we should process them from right to left
    #    to affect the least significant character first.
    
    smaller_first = [i for i in diff_indices if T[i] < S[i]]
    larger_last = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort smaller_first ascending (left to right)
    # Sort larger_last descending (right to left)
    order = sorted(smaller_first) + sorted(larger_last, reverse=True)
    
    # Use reduce to simulate the transformation process.
    # The accumulator is a list of strings (X).
    # The current state of S is tracked by the last element of the accumulator.
    def transform(acc, idx):
        current_s = acc[-1] if acc else S
        # Create a new string by changing character at idx to T[idx]
        # Using slicing to avoid loops/mutables
        next_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [next_s]

    # We start with an empty list and apply the transformation for each index in 'order'
    result_x = reduce(transform, order, [])
    
    # Output the number of elements and the elements themselves
    print(len(result_x))
    # Using join and map to avoid explicit for-loops for printing
    if result_x:
        sys.stdout.write('\n'.join(result_x) + '\n')

if __name__ == "__main__":
    solve()