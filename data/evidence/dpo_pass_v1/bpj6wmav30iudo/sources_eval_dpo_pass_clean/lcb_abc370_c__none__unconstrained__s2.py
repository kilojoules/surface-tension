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
    # To get the lexicographically smallest X, we must change characters 
    # at indices that make the resulting string smallest.
    # However, the problem asks for the minimum number of elements M.
    # M is simply the number of positions where S[i] != T[i].
    # To make X lexicographically smallest, we should prioritize changing 
    # characters at indices i where T[i] < S[i] first, and among those, 
    # the smallest index i. 
    # Wait, the lexicographical comparison is on the whole string X_j.
    # To make X_1 smallest, we want the change at the earliest possible 
    # index that results in the smallest possible string.
    # If we change index i, the string becomes S[0:i] + T[i] + S[i+1:].
    # To make this smallest, we want the smallest T[i] at the smallest i.
    # Actually, since we must eventually change all differing indices, 
    # the strategy for the lexicographically smallest array X is:
    # 1. Identify all indices i where S[i] != T[i].
    # 2. If T[i] < S[i], changing i makes the string smaller. 
    #    We should do this as early as possible (smallest i first).
    # 3. If T[i] > S[i], changing i makes the string larger. 
    #    We should delay this as long as possible (largest i first).
    
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # Indices where T[i] < S[i] (improves string) sorted ascending
    # Indices where T[i] > S[i] (worsens string) sorted descending
    order = sorted([i for i in diff_indices if T[i] < S[i]]) + \
            sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)

    # Use reduce to generate the sequence of strings
    # The accumulator is a list of strings X
    X = reduce(
        lambda acc, i: acc + [''.join(list(acc[-1]) if acc else list(S))로 
                             # This is tricky without assignment. 
                             # Let's use a helper function inside the lambda.
                             # Since we can't define functions, we use a list 
                             # to track the current state of S.
                             []], 
        order, 
        []
    )
    # Correction: reduce needs to pass the current string state.
    # We'll store (current_string, list_of_X) in the accumulator.
    
    result = reduce(
        lambda state, i: (
            (''.join([state[0][j] if j != i else T[j] for j in range(len(S))]), 
             state[1] + [''.join([state[0][j] if j != i else T[j] for j in range(len(S))])])
        ),
        order,
        (S, [])
    )

    # Output the results
    # Use print with join to avoid for-loops
    sys.stdout.write(f"{len(result[1])}\n" + "\n".join(result[1]) + ("\n" if result[1] else ""))

if __name__ == "__main__":
    solve()