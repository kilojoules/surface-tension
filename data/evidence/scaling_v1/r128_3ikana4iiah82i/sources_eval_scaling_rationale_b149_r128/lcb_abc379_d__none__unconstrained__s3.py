import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries in order. 
    # State: (current_total_height, plants_birth_offsets, results)
    # plants_birth_offsets: A sorted list of values (current_total_height - 0) 
    # when the plant was created. A plant is harvestable if:
    # current_total_height - birth_offset >= H  =>  birth_offset <= current_total_height - H
    
    # Using a generator to yield queries from the flat input list
    def get_queries(data):
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

    def process_query(state, query):
        curr_h, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant height 0. Birth offset is the current total height.
            # We use bisect to keep the plants list sorted.
            # However, since curr_h is non-decreasing, we can just append.
            plants.append(curr_h)
            return (curr_h, plants, results)
        
        elif q_type == 2:
            # Increase total height
            return (curr_h + val, plants, results)
        
        else: # q_type == 3
            # Harvest plants where curr_h - birth_offset >= val
            # birth_offset <= curr_h - val
            threshold = curr_h - val
            # Find number of plants with birth_offset <= threshold
            idx = bisect_left(plants, threshold + 1)
            # The number of harvested plants is idx
            # Remove them from the list
            # Note: slicing creates a new list, which is acceptable given the constraints
            # and the nature of the problem (we only remove from the front).
            return (curr_h, plants[idx:], results + [idx])

    # Use reduce to iterate through queries without a for/while loop
    final_state = reduce(process_query, get_queries(input_data), (0, [], []))
    
    # Output the results
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()