import sys
from functools import reduce

def solve():
    # Read N and the sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = input_data[1:]

    # We use functools.reduce to iterate through the sequence A.
    # The accumulator 'acc' will be a tuple: (last_positions_dict, result_list)
    # last_positions_dict: stores {value: last_index}
    # result_list: stores the B_i values calculated so far
    
    def process(acc, item):
        last_pos_dict, results = acc
        # Use 1-based indexing for the current position
        current_idx = len(results) + 1
        val = item[0]
        
        # Determine B_i: get value from dict or -1 if not present
        b_i = last_pos_dict.get(val, -1)
        
        # Update the dictionary with the current index for the next occurrence
        # Note: we create a new dictionary or update the existing one.
        # Since we are in a reduce, we update the mutable dict and return the tuple.
        last_pos_dict[val] = current_idx
        results.append(b_i)
        
        return (last_pos_dict, results)

    # Initialize reduce with an empty dictionary and an empty list
    # We use enumerate to get (index, value) pairs, but since we need 
    # the logic inside the function, we pass A as the sequence.
    final_state = reduce(
        lambda acc, x: process(acc, x), 
        map(lambda x: (x, None), A), 
        ({}, [])
    )

    # The second element of the final tuple is the list B
    print(*(final_state[1]))

if __name__ == "__main__":
    solve()