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
    # state = (current_time, sorted_list_of_birth_times, outputs)
    # A plant planted at time 't' has height (current_time - t) at current_time.
    # Condition: height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    # Using a generator to yield queries from the flat input list
    def get_queries():
        it = iter(input_data[1:])
        for _ in range(Q):
            q_type = next(it)
            if q_type == '1':
                yield (1, 0)
            elif q_type == '2':
                yield (2, int(next(it)))
            else:
                yield (3, int(next(it)))

    def process_query(state, query):
        current_time, plants, outputs = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new flower pot. It is born at the current_time.
            # We maintain plants as a sorted list of birth times.
            # Since current_time only increases, we can just append.
            return (current_time, plants + [current_time], outputs)
        
        elif q_type == 2:
            # Increase current time by T
            return (current_time + val, plants, outputs)
        
        else:
            # Harvest plants where current_time - birth_time >= H
            # birth_time <= current_time - H
            threshold = current_time - val
            # Find index of last plant that satisfies the condition
            idx = bisect_right(plants, threshold)
            harvested_count = idx
            # Remove harvested plants (those from 0 to idx-1)
            # We use slicing to create a new list for the state
            return (current_time, plants[idx:], outputs + [str(harvested_count)])

    # Initial state: (time, plants_list, output_list)
    initial_state = (0, [], [])
    
    # reduce replaces the for/while loop to process all queries
    final_state = reduce(process_query, get_queries(), initial_state)
    
    # Print all accumulated outputs joined by newlines
    sys.stdout.write("\n".join(final_state[2]) + "\n")

if __name__ == "__main__":
    solve()