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
    # We need to change characters at these indices to reach T
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest:
    # We want the strings X_i to be as small as possible.
    # A string is smaller if its characters at earlier positions are smaller.
    # However, we can only change ONE character per step.
    # To keep X_1 smallest, we should prioritize changing characters at the 
    # earliest possible index to the target character T[i], PROVIDED that 
    # T[i] is smaller than S[i].
    # If T[i] is larger than S[i], changing it makes the string lexicographically larger.
    # Therefore, we should change all indices i where T[i] < S[i] first (in increasing order of i),
    # and then change all indices i where T[i] > S[i] last (in decreasing order of i).
    
    # Indices that make the string smaller
    smaller = [i for i in diff_indices if T[i] < S[i]]
    # Indices that make the string larger
    larger = [i for i in diff_indices if T[i] > S[i]]
    
    # Order of operations: 
    # 1. Indices where T[i] < S[i], processed from left to right (0 to N-1)
    # 2. Indices where T[i] > S[i], processed from right to left (N-1 to 0)
    # This ensures the resulting strings in X are lexicographically as small as possible.
    order = sorted(smaller) + sorted(larger, reverse=True)
    
    # Use reduce to generate the sequence of strings.
    # The accumulator stores the list of strings X.
    # The current string is tracked by looking at the last element of the accumulator.
    def transform(acc, idx):
        current_s = acc[-1] if acc else S
        # Strings are immutable, so we create a list, change the char, and join back.
        s_list = list(current_s)
        s_list[idx] = T[idx]
        return acc + ["".join(s_list)]

    X = reduce(transform, order, [])
    
    # Output the number of elements and the elements themselves
    print(len(X))
    sys.stdout.write("\n".join(X) + "\n") if X else None

if __name__ == "__main__":
    solve()