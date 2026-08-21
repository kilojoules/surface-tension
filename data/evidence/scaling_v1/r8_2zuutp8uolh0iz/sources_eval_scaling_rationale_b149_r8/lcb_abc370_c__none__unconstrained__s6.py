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
    
    # To get the lexicographically smallest array X, we should process 
    # indices based on whether changing S[i] to T[i] makes the string 
    # lexicographically smaller or larger.
    # 1. If T[i] < S[i], changing it immediately makes the string smaller.
    #    We should process these indices from left to right.
    # 2. If T[i] > S[i], changing it makes the string larger.
    #    We should process these indices from right to left to keep the 
    #    string "smaller" for as long as possible.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending, increasing indices descending
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
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