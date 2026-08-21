import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    # Q is the first element, queries are the rest
    q_count = int(input_data[0])
    queries = input_data[1:]

    # State consists of:
    # 1. sorted_plants: A sorted list of 'birth times' (relative to a global clock)
    # 2. current_time: The total T accumulated so far
    # 3. results: A list to store the answers for type 3 queries
    
    def process_query(state, query_str):
        sorted_plants, current_time, results = state
        parts = list(map(int, query_str.split()))
        q_type = parts[0]
        
        if q_type == 1:
            # Plant height is 0. Its "birth time" is the current_time.
            # We use a sorted list to keep track of plants.
            # Since we only add plants at the 'current' time, 
            # and current_time is non-decreasing, we can just append.
            sorted_plants.append(current_time)
            return (sorted_plants, current_time, results)
        
        elif q_type == 2:
            # Increase global time
            t_val = parts[1]
            return (sorted_plants, current_time + t_val, results)
        
        else: # q_type == 3
            # Plant height = current_time - birth_time
            # Harvest if height >= H  =>  current_time - birth_time >= H
            # => birth_time <= current_time - H
            h_val = parts[1]
            threshold = current_time - h_val
            
            # Find how many plants have birth_time <= threshold
            # bisect_left returns the index of the first element > threshold
            idx = bisect_left(sorted_plants, threshold + 1)
            
            # The number of harvested plants is idx
            results.append(str(idx))
            
            # Remove harvested plants from the list
            # Using slice assignment to modify the list in place
            del sorted_plants[:idx]
            
            return (sorted_plants, current_time, results)

    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(process_query, queries, ([], 0, []))
    
    # Output all results joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()