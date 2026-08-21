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
    # The index is tracked by enumerating A (starting at 1)
    
    def accumulator(state, item):
        last_seen_map, b_list = state
        idx, val = item
        
        # Get the previous position of the current value, default to -1
        prev_pos = last_seen_map.get(val, -1)
        
        # Update the map with the current index
        # Note: dictionaries are mutable, but we return a new state tuple
        last_seen_map[val] = idx
        b_list.append(prev_pos)
        
        return (last_seen_map, b_list)

    # Initialize reduce with an empty dict and an empty list
    # enumerate(a, 1) provides (1, A_1), (2, A_2), ...
    final_state = reduce(accumulator, enumerate(a, 1), ({}, []))
    
    # The result is the second element of the final state tuple
    result = final_state[1]
    
    # Print the result joined by spaces
    sys.stdout.write(" ".join(map(str, result)) + "\n")

if __name__ == "__main__":
    solve()