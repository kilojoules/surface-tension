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
    #    We want this to happen as early as possible.
    # 2. If T[i] > S[i], changing S[i] to T[i] makes the string larger. 
    #    We want this to happen as late as possible.
    # Therefore, we process indices i where T[i] < S[i] in increasing order of i,
    # followed by indices i where T[i] > S[i] in decreasing order of i.
    
    decreasing_indices = sorted([i for i in diff_indices if T[i] < S[i]])
    increasing_indices = sorted([i for i in diff_indices if T[i] > S[i]], reverse=True)
    
    # The order of operations is: all T[i] < S[i] (left to right), then all T[i] > S[i] (right to left)
    # Wait, the logic for T[i] > S[i] is: to keep the string lexicographically small,
    # we want the "increase" to happen at the rightmost possible position first.
    # So for T[i] > S[i], we process indices in descending order.
    
    order = decreasing_indices + increasing_indices
    
    # Use reduce to generate the sequence of strings.
    # The accumulator stores the current string and the list of strings generated so far.
    def apply_change(acc, idx):
        current_s, history = acc
        # Create new string by replacing character at idx
        new_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return (new_s, history + [new_s])

    final_state = reduce(apply_change, order, (S, []))
    X = final_state[1]
    
    # Output the number of elements and the elements themselves
    sys.stdout.write(f"{len(X)}\n")
    if X:
        sys.stdout.write("\n".join(X) + "\n")

if __name__ == "__main__":
    solve()