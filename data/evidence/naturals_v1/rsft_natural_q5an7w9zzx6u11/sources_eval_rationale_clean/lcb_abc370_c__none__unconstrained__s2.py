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
    # 3. For indices that make the string smaller, we process them in increasing order of index.
    # 4. For indices that make the string larger, we process them in decreasing order of index.
    
    # Sort indices: those where T[i] < S[i] come first (ascending), 
    # then those where T[i] > S[i] (descending).
    # We use a tuple as a sort key: (is_increasing, index)
    # is_increasing is 0 if T[i] < S[i] else 1.
    # If is_increasing is 0, we want index ascending.
    # If is_increasing is 1, we want index descending.
    
    sorted_indices = sorted(
        diff_indices, 
        key=lambda i: (0, i) if T[i] < S[i] else (1, -i)
    )

    # Use reduce to generate the sequence of strings.
    # The accumulator stores the current string and the list of strings generated so far.
    def apply_change(acc, idx):
        current_s, history = acc
        # Create the new string by replacing character at idx
        new_s = current_s[:idx] + T[idx] + current_s[idx+1:]
        return (new_s, history + [new_s])

    result = reduce(apply_change, sorted_indices, (S, []))
    final_history = result[1]

    # Output the number of elements and the elements themselves
    sys.stdout.write(f"{len(final_history)}\n")
    if final_history:
        sys.stdout.write("\n".join(final_history) + "\n")

if __name__ == "__main__":
    solve()