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
    # Since we cannot use loops, we use reduce to maintain state.
    # State: (current_total_time, sorted_list_of_planting_times, results_list)
    # A plant planted at time 't' has height (current_total_time - t) at the current moment.
    # Harvest condition: height >= H  =>  current_total_time - t >= H  =>  t <= current_total_time - H.
    
    # To handle the input stream without a loop, we group the flat list into queries.
    # However, queries have variable lengths (1, 2, or 3 arguments).
    # We will use a generator or a custom function to parse the flat list.
    
    def get_queries(data):
        it = iter(data[1:])
        def parse():
            try:
                q_type = next(it)
                if q_type == '1':
                    return (1, None)
                elif q_type == '2':
                    return (2, int(next(it)))
                else:
                    return (3, int(next(it)))
            except StopIteration:
                return None
        
        # Since we can't use a while loop to call parse(), 
        # we use a recursive-like approach via map/filter or a list comprehension
        # But the number of elements consumed depends on the type.
        # Let's use a helper function with reduce to process the flat list.
        return it

    def process_queries(state, query_tuple):
        current_time, plants, results = state
        q_type, val = query_tuple
        
        if q_type == 1:
            # Plant a new flower. Its "birth time" is the current_time.
            # We maintain the plants list sorted. Since current_time is non-decreasing,
            # we can just append.
            return (current_time, plants + [current_time], results)
        
        elif q_type == 2:
            # Increase time by T.
            return (current_time + val, plants, results)
        
        else: # q_type == 3
            # Harvest plants where current_time - t >= H  => t <= current_time - val
            threshold = current_time - val
            # Find index of first plant with t > threshold
            idx = bisect_left(plants, threshold + 1)
            harvested_count = idx
            # Remaining plants are those from idx onwards
            return (current_time, plants[idx:], results + [harvested_count])

    # Because the input is variable length, we can't use a simple map.
    # We use a custom reducer to handle the flat input list.
    def flat_reducer(state, item):
        # state: (current_time, plants, results, current_query_type, pending_val)
        curr_t, plants, res, q_type, p_val = state
        
        if q_type is None:
            # Starting a new query
            t = item
            if t == '1':
                # Type 1 is complete immediately
                return (curr_t, plants + [curr_t], res + [0], None, None)
            else:
                # Type 2 or 3 needs one more value
                return (curr_t, plants, res, t, None)
        else:
            # Completing a query of type 2 or 3
            val = int(item)
            if q_type == '2':
                return (curr_t + val, plants, res, None, None)
            else: # Type 3
                threshold = curr_t - val
                idx = bisect_left(plants, threshold + 1)
                return (curr_t, plants[idx:], res + [idx], None, None)

    # The flat_reducer logic above is slightly different from process_queries.
    # Let's refine it to handle the state correctly.
    def final_reducer(state, item):
        curr_t, plants, res, q_type = state
        if q_type == 0: # Waiting for query type
            if item == '1':
                return (curr_t, plants + [curr_t], res + [0], 0)
            elif item == '2':
                return (curr_t, plants, res, 2)
            else:
                return (curr_t, plants, res, 3)
        elif q_type == 2: # Waiting for T
            return (curr_t + int(item), plants, res, 0)
        else: # q_type == 3, waiting for H
            threshold = curr_t - int(item)
            idx = bisect_left(plants, threshold + 1)
            return (curr_t, plants[idx:], res + [idx], 0)

    # Initial state: (current_time, plants_list, results_list, query_type_state)
    initial_state = (0, [], [], 0)
    final_state = reduce(final_reducer, input_data[1:], initial_state)
    
    # The results list contains 0s for type 1 and 2, and actual counts for type 3.
    # However, the logic above adds 0 for type 1. We need to filter those.
    # Actually, the prompt says output K lines for K queries of type 3.
    # Let's adjust the reducer to only add to results on type 3.
    
    def refined_reducer(state, item):
        curr_t, plants, res, q_type = state
        if q_type == 0:
            if item == '1':
                return (curr_t, plants + [curr_t], res, 0)
            elif item == '2':
                return (curr_t, plants, res, 2)
            else:
                return (curr_t, plants, res, 3)
        elif q_type == 2:
            return (curr_t + int(item), plants, res, 0)
        else:
            threshold = curr_t - int(item)
            idx = bisect_left(plants, threshold + 1)
            return (curr_t, plants[idx:], res + [idx], 0)

    final_state = reduce(refined_reducer, input_data[1:], (0, [], [], 0))
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()