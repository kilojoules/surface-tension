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
    
    # To get the lexicographically smallest array X, we must change characters
    # at indices that make the resulting string as small as possible.
    # However, we can only change one character per step.
    # To keep the string lexicographically smallest at each step:
    # 1. If we change S[i] to T[i] and T[i] < S[i], we want to do this as early as possible.
    # 2. If we change S[i] to T[i] and T[i] > S[i], we want to do this as late as possible.
    
    # Divide indices into those that decrease the string and those that increase it.
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (left to right) to reduce the string ASAP.
    # Sort increasing indices descending (right to left) to increase the string as late as possible.
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a list of strings (the array X).
    def evolve(acc, idx):
        current_s = acc[-1] if acc else S
        # Create new string by replacing character at idx with T[idx]
        next_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return acc + [next_s]

    X = reduce(evolve, order, [])
    
    # Output the number of elements and the elements themselves
    print(len(X))
    sys.stdout.write('\n'.join(X) + ('\n' if X else ''))

if __name__ == "__main__":
    solve()