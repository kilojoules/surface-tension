import sys
from bisect import bisect_left
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # State structure: (current_time, plants_birth_offsets, output_buffer)
    # plants_birth_offsets is a sorted list of times when plants were added.
    # A plant added at time 't' has height (current_time - t).
    # Height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    def process_queries(state, query_bundle):
        current_time, plants, outputs = state
        q_type = query_bundle[0]
        
        if q_type == '1':
            # Plant a new plant at the current global time
            # We use a sorted list to keep track of birth times
            # Since we only add plants at the 'current' time, 
            # and current_time is non-decreasing, the list remains sorted.
            return (current_time, plants + [current_time], outputs)
        
        elif q_type == '2':
            # Increase global timer
            t_val = int(query_bundle[1])
            return (current_time + t_val, plants, outputs)
        
        elif q_type == '3':
            # Harvest plants where birth_time <= current_time - H
            h_val = int(query_bundle[1])
            threshold = current_time - h_val
            
            # Find how many plants have birth_time <= threshold
            # bisect_left finds the index of the first element > threshold
            idx = bisect_left(plants, threshold + 1) 
            # Wait, the condition is height >= H, so birth_time <= current_time - H.
            # We need the count of elements in 'plants' that are <= threshold.
            # bisect_right is better for <= threshold.
            from bisect import bisect_right
            idx = bisect_right(plants, threshold)
            
            # The number of harvested plants is idx.
            # The remaining plants are those from index idx onwards.
            return (current_time, plants[idx:], outputs + [str(idx)])
            
    # Since we cannot use loops, we group the flat input list into queries.
    # This is tricky because queries have different lengths.
    # We'll use a helper function to parse the queries into a list of tuples first.
    
    def parse_queries(data):
        # We use a recursive-like approach via a generator or a custom reducer 
        # to group the input based on the query type.
        def grouper(acc, item):
            # acc: (current_group, all_groups)
            curr, all_g = acc
            if not curr:
                # Start a new group with the type
                return ([item], [])
            # This logic is flawed for variable lengths. 
            # Let's use a different approach to parse.
            pass

    # Correct parsing: since we can't use while/for, we use a 
    # custom function with reduce to partition the flat list.
    def get_query_batches(data):
        def step(state, _):
            # state: (remaining_data, batches)
            rem, batches = state
            if not rem: return (rem, batches)
            q_type = rem[0]
            if q_type == '1':
                return (rem[1:], batches + [('1',)])
            elif q_type == '2':
                return (rem[2:], batches + [('2', rem[1])])
            else:
                return (rem[2:], batches + [('3', rem[1])])
        
        # We need to run this Q times. 
        # We can use range(Q) and reduce to simulate the loop.
        return reduce(step, range(Q), (data, []))[1]

    batches = get_query_batches(queries)
    
    # Now process the batches
    # Note: we redefine bisect_right inside the reducer logic via a helper
    def final_reducer(state, q):
        current_time, plants, outputs = state
        if q[0] == '1':
            return (current_time, plants + [current_time], outputs)
        elif q[0] == '2':
            return (current_time + int(q[1]), plants, outputs)
        else:
            from bisect import bisect_right
            threshold = current_time - int(q[1])
            idx = bisect_right(plants, threshold)
            return (current_time, plants[idx:], outputs + [str(idx)])

    final_state = reduce(final_reducer, batches, (0, [], []))
    sys.stdout.write('\n'.join(final_state[2]) + '\n')

if __name__ == "__main__":
    solve()