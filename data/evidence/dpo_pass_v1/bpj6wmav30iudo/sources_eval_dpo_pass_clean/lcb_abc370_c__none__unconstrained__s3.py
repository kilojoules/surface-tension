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
    # We need to change S[i] to T[i] for all i where S[i] != T[i]
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    
    # To make the array X lexicographically smallest, we want the strings 
    # at the beginning of X to be as small as possible.
    # This means we should prioritize changing characters at the 
    # leftmost positions to their target values T[i] ONLY IF T[i] < S[i].
    # If T[i] > S[i], we want to delay that change as long as possible 
    # to keep the string lexicographically smaller.
    
    # Correct strategy for lexicographical minimality:
    # 1. Change all i where T[i] < S[i] from left to right.
    # 2. Change all i where T[i] > S[i] from right to left.
    
    left_to_right = [i for i in diff_indices if T[i] < S[i]]
    right_to_left = [i for i in diff_indices if T[i] > S[i]][::-1]
    
    # The sequence of indices to change
    change_order = left_to_right + right_to_left
    
    # Use reduce to generate the sequence of strings.
    # The accumulator is a tuple: (current_string, list_of_strings_X)
    def apply_change(acc, idx):
        current_s, x_list = acc
        # Create new string by replacing character at idx
        # Using a list comprehension to simulate string mutation
        new_s = "".join([current_s[i] if i != idx else T[i] for i in range(len(current_s))])
        return (new_s, x_list + [new_s])

    result = reduce(apply_change, change_order, (S, []))
    m, x = result[1], len(result[1])
    
    # Output the result
    # Using sys.stdout.write to avoid print loop
    sys.stdout.write(f"{m}\n" + "\n".join(x) + ("\n" if m > 0 else ""))

if __name__ == "__main__":
    solve()