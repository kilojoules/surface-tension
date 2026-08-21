import sys
from functools import reduce

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s_str = input_data[0]
    t_str = input_data[1]
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(s_str)) if s_str[i] != t_str[i]]
    
    # To get the lexicographically smallest sequence:
    # 1. First, handle indices i where T[i] < S[i] in increasing order of i.
    #    Changing a character to a smaller one immediately makes the string smaller.
    # 2. Then, handle indices i where T[i] > S[i] in decreasing order of i.
    #    Changing a character to a larger one makes the string larger, so we 
    #    want to do this as late as possible, and for the rightmost indices first.
    
    decreasing_indices = [i for i in diff_indices if t_str[i] < s_str[i]]
    increasing_indices = [i for i in diff_indices if t_str[i] > s_str[i]][::-1]
    
    # The sequence of indices to change
    order = decreasing_indices + increasing_indices
    
    # Use reduce to simulate the transformation of S and collect the resulting strings.
    # accumulator is a tuple: (current_string_list, sequence_of_strings_X)
    def transform(acc, idx):
        current_s, x_array = acc
        # Create a new list for the string to avoid mutating the previous state
        new_s = list(current_s)
        new_s[idx] = t_str[idx]
        res_s = "".join(new_s)
        return (new_s, x_array + [res_s])

    initial_state = (list(s_str), [])
    final_state = reduce(transform, order, initial_state)
    
    x_result = final_state[1]
    
    # Output the number of elements and the elements themselves
    print(len(x_result))
    # Using join and map to avoid a for-loop for printing
    if x_result:
        sys.stdout.write("\n".join(x_result) + "\n")

if __name__ == "__main__":
    solve()