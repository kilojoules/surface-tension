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
    # Each query is represented as a tuple/list.
    # Since query lengths vary, we'll use a generator to yield them.
    def get_queries(data):
        it = iter(data[1:])
        def next_q():
            try:
                q_type = next(it)
                if q_type == '1':
                    return (1, 0)
                elif q_type == '2':
                    return (2, int(next(it)))
                else:
                    return (3, int(next(it)))
            except StopIteration:
                return None
        
        # Create a list of all queries
        res = []
        while True:
            q = next_q()
            if q is None: break
            res.append(q)
        return res

    queries = get_queries(input_data)

    # State: (plants_sorted_list, current_time_offset, output_list)
    # A plant planted at time 't' has height (current_time - t).
    # It is harvested if (current_time - t) >= H  =>  t <= current_time - H.
    # We store the 'birth time' of plants in a sorted list.
    
    def process_query(state, query):
        plants, current_time, outputs = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new plant at the current time offset.
            # Since current_time only increases, we can just append.
            # However, to keep it strictly loop-free and functional, 
            # we create a new list.
            return (plants + [current_time], current_time, outputs)
        
        elif q_type == 2:
            # Increase current time
            return (plants, current_time + val, outputs)
        
        else:
            # Harvest plants with height >= H.
            # Height = current_time - birth_time.
            # current_time - birth_time >= H  => birth_time <= current_time - H.
            threshold = current_time - val
            # Find how many plants have birth_time <= threshold.
            # bisect_left returns the index of the first element > threshold.
            idx = bisect_left(plants, threshold + 1) 
            # Wait, birth_time is an integer. birth_time <= threshold 
            # is equivalent to birth_time < threshold + 1.
            # Actually, bisect_right is better for <=.
            from bisect import bisect_right
            idx = bisect_right(plants, threshold)
            
            # The number of harvested plants is idx.
            # The remaining plants are those from index idx onwards.
            return (plants[idx:], current_time, outputs + [str(idx)])

    # Using reduce to simulate the state machine
    final_state = reduce(process_query, queries, ([], 0, []))
    
    # Print all harvested counts
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()