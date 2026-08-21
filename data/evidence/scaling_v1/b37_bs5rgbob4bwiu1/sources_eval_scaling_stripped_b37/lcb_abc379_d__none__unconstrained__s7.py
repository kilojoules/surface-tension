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
    
    # We need to process the queries to extract the ones we care about.
    # Since we cannot use loops, we use a generator/map to parse the input.
    # The input format is irregular (1, 2 T, 3 H), so we use a custom 
    # function to group the flat list into queries.
    
    def group_queries(data):
        it = iter(data[1:])
        def next_query():
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
        
        # Use map to create a sequence of queries until None is encountered
        # Since map doesn't have a 'while' condition, we use a range and 
        # a helper function to maintain state via a list/closure.
        return map(lambda _: next_query(), range(Q))

    # To avoid loops and maintain state, we use reduce.
    # State structure: (current_total_time, sorted_list_of_planting_times, results_list)
    # A plant planted at time 't' has height (current_total_time - t) at the current moment.
    # Condition: height >= H  =>  current_total_time - t >= H  =>  t <= current_total_time - H.
    
    def process(state, query):
        current_time, planting_times, results = state
        q_type, val = query
        
        if q_type == 1:
            # Plant a new seed at the current timestamp
            # We use sorted() to keep the list of planting times sorted
            # Since we only add one element, we can use bisect or just sorted()
            # But for efficiency in reduce, we maintain the sorted property.
            # We use a list comprehension to insert the value in the correct spot.
            # Actually, since current_time is non-decreasing, new plants are always 
            # added to the end.
            return (current_time, planting_times + [current_time], results)
        
        elif q_type == 2:
            # Increase global time
            return (current_time + val, planting_times, results)
        
        else:
            # Harvest plants where planting_time <= current_time - val
            threshold = current_time - val
            # Find index of first plant that is NOT harvested
            idx = bisect_left(planting_times, threshold + 1)
            # Wait, the condition is height >= H, so planting_time <= current_time - H.
            # Plants at indices [0, idx-1] are harvested.
            # Correct index for bisect: find first element > (current_time - val)
            # Actually, bisect_right finds the index after the last element <= threshold.
            import bisect
            idx = bisect.bisect_right(planting_times, threshold)
            
            harvested_count = idx
            remaining_plants = planting_times[idx:]
            return (current_time, remaining_plants, results + [harvested_count])

    # Initialize state: (total_time, planting_times_list, results_list)
    initial_state = (0, [], [])
    
    # We wrap the query grouping in a list to ensure it's fully consumed by reduce
    queries = list(group_queries(input_data))
    
    final_state = reduce(process, queries, initial_state)
    
    # Print all results joined by newlines
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()