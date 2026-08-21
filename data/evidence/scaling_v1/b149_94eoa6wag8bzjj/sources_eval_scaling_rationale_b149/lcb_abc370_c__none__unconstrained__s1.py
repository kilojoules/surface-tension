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
    # 1. We want the strings in X to be as small as possible as early as possible.
    # 2. If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    #    We should do this for the leftmost indices first.
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    We should do this for the rightmost indices first to keep the prefix small for longer.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Order: indices where T[i] < S[i] (left to right), then T[i] > S[i] (right to left)
    order = decreasing + increasing[::-1]

    # Use reduce to generate the sequence of strings
    # The accumulator is a list of strings (the array X)
    def evolve(acc, idx):
        current_s = acc[-1] if acc else S
        # Create new string by replacing character at idx
        next_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [next_s]

    X = reduce(evolve, order, [])

    # Output the result
    print(len(X))
    if X:
        print('\n'.join(X))

if __name__ == "__main__":
    solve()