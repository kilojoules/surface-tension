import sys
from bisect import bisect_right
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    
    # We need to process queries and maintain state:
    # state = (current_time, sorted_list_of_planting_times, results_list)
    # A plant planted at time 't' has height (current_time - t) at current_time.
    # Height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    # To handle the input stream, we group the flat list into queries.
    # Since queries have different lengths, we use a generator or a custom parser.
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

    queries = list(get_queries(input_data))

    def process(state, query):
        current_time, plants, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new flower pot at the current time
            # We use a sorted list to keep track of planting times
            # Since we always add the current_time, and current_time is non-decreasing,
            # the list remains sorted.
            return (current_time, plants + [current_time], results)
        
        elif q_type == 2:
            # Increase current time by T
            return (current_time + val, plants, results)
        
        else: # q_type == 3
            # Harvest plants where current_time - t >= H  => t <= current_time - H
            threshold = current_time - val
            # Find index of last plant that satisfies t <= threshold
            idx = bisect_right(plants, threshold)
            harvested_count = idx
            # Remove harvested plants (those from 0 to idx-1)
            return (current_time, plants[idx:], results + [harvested_count])

    # Initial state: (time, plants_list, results_list)
    initial_state = (0, [], [])
    final_state = reduce(process, queries, initial_state)
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()