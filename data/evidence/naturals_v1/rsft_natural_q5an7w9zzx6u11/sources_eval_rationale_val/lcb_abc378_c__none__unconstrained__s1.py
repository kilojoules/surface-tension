import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, A is the rest
    n = int(input_data[0])
    a = input_data[1:]
    
    # We use reduce to maintain a state consisting of:
    # 1. A dictionary mapping value -> last seen index
    # 2. A list containing the resulting B values
    # The index i is tracked using enumerate(a, 1)
    
    def accumulator(state, item):
        last_seen_map, b_list = state
        i, val = item
        
        # Get the previous index of the current value, default to -1
        prev_idx = last_seen_map.get(val, -1)
        
        # Update the map with the current index
        last_seen_map[val] = i
        
        # Append the result to the list
        b_list.append(prev_idx)
        
        return (last_seen_map, b_list)

    # Initial state: empty dict and empty list
    # We pass enumerate(a, 1) to reduce to get 1-based indexing
    final_state = reduce(accumulator, enumerate(a, 1), ({}, []))
    
    # The result is the second element of the final state tuple
    print(*(final_state[1]))

if __name__ == "__main__":
    solve()