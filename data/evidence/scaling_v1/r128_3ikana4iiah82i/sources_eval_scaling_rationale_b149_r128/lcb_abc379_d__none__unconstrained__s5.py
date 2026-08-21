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
    # State: (current_total_height, plants_offsets, results)
    # plants_offsets: a sorted list of values (current_total_height - plant_height_at_birth)
    # A plant is harvested if: current_total_height - offset >= H  => offset <= current_total_height - H
    
    # Using a generator to yield queries based on their type
    def get_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, None)
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
            # Plant height 0. Offset = curr_h - 0 = curr_h.
            # We use bisect to keep the offsets list sorted.
            # However, since curr_h is non-decreasing, we can just append.
            plants.append(curr_h)
            return (curr_h, plants, results)
        
        elif q_type == 2:
            # Increase total height
            return (curr_h + val, plants, results)
        
        elif q_type == 3:
            # Harvest plants where curr_h - offset >= val  => offset <= curr_h - val
            threshold = curr_h - val
            # Find number of plants with offset <= threshold
            idx = bisect_left(plants, threshold + 1) 
            # Note: bisect_left(plants, threshold + 1) gives count of elements <= threshold
            
            # The number of harvested plants is idx
            # Remove them from the list
            # Since we cannot use loops, we use slice assignment or a new list
            # But we must avoid explicit loops. 
            # We can use the fact that we are removing a prefix of the sorted list.
            harvested_count = idx
            # Update plants list by removing the first 'idx' elements
            # Using slice notation to modify the list in place
            del plants[:idx]
            
            results.append(str(harvested_count))
            return (curr_h, plants, results)

    # Use reduce to process all queries
    final_state = reduce(process_query, get_queries(input_data), (0, [], []))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()