import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, A is the rest
    N = int(input_data[0])
    A = input_data[1:]
    
    # We use functools.reduce to iterate through the sequence A.
    # The accumulator stores a tuple: (last_positions_dict, result_list)
    # last_positions_dict: maps the value A_i to its most recent 1-based index.
    # result_list: the sequence B being constructed.
    
    # We use enumerate(A, 1) to get 1-based indexing for the positions.
    def accumulate_positions(state, item):
        idx, val = item
        last_pos_dict, res = state
        
        # Get the previous position of the current value, default to -1
        prev_pos = last_pos_dict.get(val, -1)
        
        # Update the dictionary with the current index for future occurrences
        # Note: we create a new dictionary reference or update the existing one.
        # Since we are in a reduce, updating the dict in-place is efficient.
        last_pos_dict[val] = idx
        
        # Append the result to the list
        res.append(prev_pos)
        
        return (last_pos_dict, res)

    # Initial state: empty dictionary and empty result list
    initial_state = ({}, [])
    
    # Perform the reduction
    final_state = reduce(accumulate_positions, enumerate(A, 1), initial_state)
    
    # The result B is the second element of the final state tuple
    B = final_state[1]
    
    # Print the result as a space-separated string
    sys.stdout.write(" ".join(map(str, B)) + "\n")

if __name__ == "__main__":
    solve()