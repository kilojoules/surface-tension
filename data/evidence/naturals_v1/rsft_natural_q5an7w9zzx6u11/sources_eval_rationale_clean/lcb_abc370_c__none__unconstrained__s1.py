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
    # 1. We want the strings in X to be as small as possible.
    # 2. If S[i] > T[i], changing S[i] to T[i] makes the string smaller.
    #    We should do this as early as possible, and for the leftmost index first.
    # 3. If S[i] < T[i], changing S[i] to T[i] makes the string larger.
    #    We should do this as late as possible, and for the rightmost index first.
    
    # Sort indices: 
    # Primary key: whether S[i] > T[i] (True comes after False, so we use a custom key)
    # Secondary key: index i (ascending for S[i] > T[i], descending for S[i] < T[i])
    
    # We split indices into two groups:
    # Group A: S[i] > T[i] (Decrease) - Process from left to right
    # Group B: S[i] < T[i] (Increase) - Process from right to left
    
    group_a = sorted([i for i in diff_indices if S[i] > T[i]])
    group_b = sorted([i for i in diff_indices if S[i] < T[i]], reverse=True)
    
    order = group_a + group_b

    # Use reduce to generate the sequence of strings.
    # The accumulator is a tuple (current_string, history_list).
    def transform(acc, idx):
        curr_s, history = acc
        # Strings are immutable, so we create a new one
        next_s = curr_s[:idx] + T[idx] + curr_s[idx+1:]
        return (next_s, history + [next_s])

    result = reduce(transform, order, (S, []))
    history = result[1]

    # Output the number of elements and the elements themselves
    sys.stdout.write(f"{len(history)}\n")
    if history:
        sys.stdout.write("\n".join(history) + "\n")

if __name__ == "__main__":
    solve()