import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, A is the rest
    n = int(input_data[0])
    a = input_data[1:]
    
    # We use reduce to process the sequence A.
    # The accumulator stores a tuple: (last_seen_dict, result_list)
    # last_seen_dict: maps the value A_i to its 1-based index
    # result_list: stores the B_i values calculated so far
    
    # We use enumerate(a, 1) to get 1-based indexing for i
    def accumulate_positions(acc, item):
        i, val = item
        last_seen, results = acc
        
        # Determine B_i: get value from dict or -1 if not present
        b_i = last_seen.get(val, -1)
        
        # Update the dictionary with the current index for the next occurrence
        # Note: dictionaries are mutable, but to stay strictly functional 
        # we update and return the same reference.
        last_seen[val] = i
        
        # Append the result to the list
        results.append(b_i)
        
        return (last_seen, results)

    # Initialize reduce with an empty dict and an empty list
    final_state = reduce(accumulate_positions, enumerate(a, 1), ({}, []))
    
    # The second element of the final state tuple is our sequence B
    print(*(final_state[1]))

if __name__ == "__main__":
    solve()