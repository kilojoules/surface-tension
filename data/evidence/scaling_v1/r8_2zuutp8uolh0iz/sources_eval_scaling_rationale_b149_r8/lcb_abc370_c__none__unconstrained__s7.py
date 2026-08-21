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
    #    These should be done first, ordered by index i (left to right).
    # 3. If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    #    These should be done last, ordered by index i (right to left).
    
    smaller_first = [i for i in diff_indices if T[i] < S[i]]
    larger_last = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort smaller_first ascending (left to right)
    # Sort larger_last descending (right to left)
    order = sorted(smaller_first) + sorted(larger_last, reverse=True)

    # Use reduce to generate the sequence of strings.
    # The accumulator is a tuple (current_string, history_list).
    def apply_change(acc, idx):
        curr_s, history = acc
        # Create new string by replacing character at idx
        new_s = curr_s[:idx] + T[idx] + curr_s[idx+1:]
        return (new_s, history + [new_s])

    initial_state = (S, [])
    final_state = reduce(apply_change, order, initial_state)
    
    X = final_state[1]
    
    # Output the result
    sys.stdout.write(f"{len(X)}\n")
    sys.stdout.write("\n".join(X) + ("\n" if X else ""))

if __name__ == "__main__":
    solve()