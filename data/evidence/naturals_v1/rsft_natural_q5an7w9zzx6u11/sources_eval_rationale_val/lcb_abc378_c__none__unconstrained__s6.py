import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N = int(input_data[0])
    A = input_data[1:]
    
    # We use reduce to maintain a state consisting of:
    # 1. A dictionary mapping value -> last seen index (1-indexed)
    # 2. A list containing the B_i values calculated so far
    # The lambda takes the accumulator (last_seen, results) and the current item (val, idx)
    
    # Use enumerate(A, 1) to get 1-based indexing
    initial_state = ({}, [])
    
    def accumulate_indices(state, item):
        val, idx = item
        last_seen, results = state
        
        # Determine B_i: the previous index of val or -1
        b_i = last_seen.get(val, -1)
        
        # Update the dictionary with the current index for future occurrences
        # Note: we create a new dictionary reference or update the existing one.
        # Since we cannot use loops, we update the dictionary and append to the list.
        last_seen[val] = idx
        results.append(b_i)
        
        return (last_seen, results)

    # Process the sequence
    final_state = reduce(accumulate_indices, enumerate(A, 1), initial_state)
    
    # Extract the results list and print as space-separated strings
    print(*(final_state[1]))

if __name__ == "__main__":
    solve()