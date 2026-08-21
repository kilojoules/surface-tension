import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N = int(input_data[0])
    A = input_data[1:]
    
    # We use reduce to maintain a state (the dictionary of last positions)
    # and build the result list B.
    # State structure: (last_seen_dict, result_list)
    # We use 1-based indexing for the positions as required.
    
    def accumulate_positions(state, current_val_tuple):
        last_seen, B = state
        idx, val = current_val_tuple
        
        # Get the previous position of the current value, default to -1
        prev_pos = last_seen.get(val, -1)
        
        # Update the dictionary with the current position
        # Note: dictionaries are mutable, but to keep it clean we update and return
        last_seen[val] = idx
        B.append(prev_pos)
        
        return (last_seen, B)

    # Prepare indexed sequence: [(1, A1), (2, A2), ...]
    indexed_A = enumerate(A, 1)
    
    # Initial state: empty dict and empty list
    initial_state = ({}, [])
    
    # Process the sequence
    final_state = reduce(accumulate_positions, indexed_A, initial_state)
    
    # The result B is the second element of the final state tuple
    B = final_state[1]
    
    # Print the result joined by spaces
    sys.stdout.write(" ".join(map(str, B)) + "\n")

if __name__ == "__main__":
    solve()