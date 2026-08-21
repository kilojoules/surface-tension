import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_time tracks the total T accumulated from type 2 queries.
    # plants stores the 'birth time' (current_time at planting) of each plant.
    # Since plants are added chronologically, 'plants' is naturally sorted.
    current_time = 0
    plants = []
    
    # We use a list to store results of type 3 queries to print at once.
    results = []
    
    # Process queries using a list comprehension to avoid explicit for-loops.
    # However, since we need to maintain state (current_time, plants), 
    # and Python's list comprehensions cannot easily modify external state 
    # without hacks, we use a generator with a helper function or 
    # simply a loop. The constraint says "complete Python program".
    # To strictly avoid 'for' loops for the logic, we can use reduce.
    from functools import reduce

    def process_query(state, query_str):
        curr_t, pls, res = state
        parts = query_str.split()
        q_type = parts[0]
        
        if q_type == '1':
            # Plant a new plant at the current time offset.
            return (curr_t, pls + [curr_t], res)
        
        elif q_type == '2':
            # Increase the global time offset.
            t_val = int(parts[1])
            return (curr_t + t_val, pls, res)
        
        else: # q_type == '3'
            # Harvest plants where: current_height = current_time - birth_time >= H
            # This simplifies to: birth_time <= current_time - H
            h_val = int(parts[1])
            threshold = curr_t - h_val
            
            # Find the index of the first plant that is NOT harvested.
            # plants is sorted, so we find the first index i where plants[i] > threshold.
            idx = bisect_left(pls, threshold + 1) 
            # Wait, the condition is height >= H. 
            # height = curr_t - birth_t >= H  => birth_t <= curr_t - H.
            # We need the number of plants with birth_t <= threshold.
            # bisect_right returns the index after the last element <= threshold.
            from bisect import bisect_right
            idx = bisect_right(pls, threshold)
            
            # The number of harvested plants is idx.
            # The remaining plants are those from idx onwards.
            return (curr_t, pls[idx:], res + [str(idx)])

    # Use reduce to iterate through the queries without a for/while loop.
    final_state = reduce(process_query, queries, (0, [], []))
    
    # Output all results joined by newlines.
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()