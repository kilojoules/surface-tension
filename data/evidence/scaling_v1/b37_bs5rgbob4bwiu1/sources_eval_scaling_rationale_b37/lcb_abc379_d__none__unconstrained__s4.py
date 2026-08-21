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
    # State: (current_time_offset, sorted_list_of_plant_birth_times, results_list)
    # A plant planted at total_time 'S' has height (current_time - S) at current_time.
    # Harvest condition: current_time - S >= H  =>  S <= current_time - H.
    
    # Parse queries into a list of tuples
    # Since queries have different lengths, we use a generator or a custom parser.
    def parse_queries(data, idx):
        if idx >= len(data):
            return []
        q_type = data[idx]
        if q_type == '1':
            return [('1',)] + parse_queries(data, idx + 1)
        elif q_type == '2':
            return [('2', int(data[idx+1]))] + parse_queries(data, idx + 2)
        else:
            return [('3', int(data[idx+1]))] + parse_queries(data, idx + 2)

    # To avoid recursion depth issues with parse_queries, we use a while loop 
    # to group the input into queries first, then use reduce.
    queries = []
    i = 1
    while i < len(input_data):
        t = input_data[i]
        if t == '1':
            queries.append(('1',))
            i += 1
        elif t == '2':
            queries.append(('2', int(input_data[i+1])))
            i += 2
        else:
            queries.append(('3', int(input_data[i+1])))
            i += 2

    def process(state, query):
        current_time, plants, results = state
        q_type = query[0]
        
        if q_type == '1':
            # Plant height 0 means it is born at the current_time
            # We keep plants sorted by birth time to use binary search
            # Since we always add the current_time, the list remains sorted
            return (current_time, plants + [current_time], results)
        
        elif q_type == '2':
            # Increase global time
            return (current_time + query[1], plants, results)
        
        else:
            # Harvest plants where current_time - birth_time >= H
            # birth_time <= current_time - H
            h_val = query[1]
            threshold = current_time - h_val
            # Find index of first plant born AFTER the threshold
            idx = bisect_left(plants, threshold + 1)
            # Plants from 0 to idx-1 are harvested
            harvested_count = idx
            # Remove harvested plants from the list
            return (current_time, plants[idx:], results + [harvested_count])

    # Using reduce to simulate the loop over queries
    # Initial state: (time=0, plants=[], results=[])
    final_state = reduce(process, queries, (0, [], []))
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()