import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and maintain state:
    # state = (current_total_height, sorted_list_of_birth_heights, results_list)
    # A plant planted at total_height 'S' has current height (current_total_height - S).
    # Condition: current_total_height - S >= H  =>  S <= current_total_height - H.
    
    # Using a generator to yield queries from the flat input list
    def get_queries(data):
        it = iter(data[1:])
        while True:
            try:
                q_type = next(it)
                if q_type == '1':
                    yield (1, None)
                elif q_type == '2':
                    yield (2, int(next(it)))
                else:
                    yield (3, int(next(it)))
            except StopIteration:
                break

    def process_query(state, query):
        total_height, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant height 0 means its "birth height" is the current total_height
            # We maintain plants as a sorted list of birth heights
            # Since total_height only increases, new plants are always added to the end
            return (total_height, plants + [total_height], results)
        
        elif q_type == 2:
            # Increase global height offset
            return (total_height + val, plants, results)
        
        else:
            # Harvest plants where total_height - birth_height >= H
            # birth_height <= total_height - H
            threshold = total_height - val
            # Find index of first plant with birth_height > threshold
            idx = bisect_left(plants, threshold + 1)
            # Plants from 0 to idx-1 are harvested
            harvested_count = idx
            # Remove harvested plants from the list
            return (total_height, plants[idx:], results + [harvested_count])

    # Use reduce to simulate the loop over queries
    # Initial state: (total_height, plants_list, results_list)
    final_state = reduce(process_query, get_queries(input_data), (0, [], []))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()