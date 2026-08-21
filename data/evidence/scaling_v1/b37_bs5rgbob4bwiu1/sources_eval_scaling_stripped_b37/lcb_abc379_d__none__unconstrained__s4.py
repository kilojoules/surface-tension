import sys
from functools import reduce
from itertools import accumulate
from bisect import bisect_left

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to parse the queries. Since they have different lengths,
    # we process the input list using a custom reducer.
    # State structure: (current_input_index, current_time, plants_birth_times, results_list)
    # plants_birth_times: sorted list of times when plants were added.
    # A plant added at time 't' has height (current_time - t) at 'current_time'.
    # Condition: height >= H  =>  (current_time - t) >= H  =>  t <= current_time - H.
    
    def process_queries(state, _):
        idx, current_time, plants, results = state
        if idx >= len(input_data) - 1:
            # This case handles the end of the input stream
            # However, the reducer runs Q times, so we manage the index manually.
            return state
            
        # This is a helper to handle the variable length of queries
        # But since we are in a reduce, we can't easily jump indices.
        # Instead, we will pre-parse the queries into a list of tuples.
        return state

    # Pre-parsing the input into a list of queries
    # Since we cannot use loops, we use a generator/map to group the input
    def parse_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, 0)
                elif q_type == '2':
                    yield (2, int(next(it)))
                elif q_type == '3':
                    yield (3, int(next(it)))
            except StopIteration:
                break

    queries = list(parse_queries(input_data))
    
    # State: (current_time, plants_birth_times, results)
    # We use a list for plants_birth_times and maintain it sorted.
    # Since we only append and remove from the left (smallest birth times),
    # and we need to count elements <= threshold, bisect_left is perfect.
    
    def reducer(state, q):
        current_time, plants, results = state
        q_type, val = q
        
        if q_type == 1:
            # Plant new flower at current_time
            # We use a list and append. Since current_time is non-decreasing,
            # the list remains sorted.
            return (current_time, plants + [current_time], results)
        
        elif q_type == 2:
            # Increase time
            return (current_time + val, plants, results)
        
        elif q_type == 3:
            # Harvest plants where current_time - birth_time >= val
            # birth_time <= current_time - val
            threshold = current_time - val
            # Find index of first plant born after the threshold
            idx = bisect_left(plants, threshold + 1)
            # Number of plants harvested is the number of plants born at or before threshold
            # We slice the list to remove harvested plants
            return (current_time, plants[idx:], results + [idx])
            
    # Initial state: (time, plants_list, results_list)
    initial_state = (0, [], [])
    
    # Process all queries
    final_state = reduce(reducer, queries, initial_state)
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()