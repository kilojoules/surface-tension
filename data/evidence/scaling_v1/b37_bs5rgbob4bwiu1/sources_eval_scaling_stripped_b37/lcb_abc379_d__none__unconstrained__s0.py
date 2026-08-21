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
    
    # We need to parse the queries. Since they have different lengths,
    # we process the input list using a custom reducer.
    # State structure: (current_index, current_time, plants_birth_times, results)
    # plants_birth_times: sorted list of times when plants were added.
    # A plant added at time 't' has height (current_time - t) at current_time.
    # Condition: height >= H  =>  current_time - t >= H  =>  t <= current_time - H.
    
    def process_queries(state, _):
        idx, current_time, plants, results = state
        if idx >= len(input_data) - 1:
            # This case handles the end of the input stream
            # However, the reducer runs Q times, so we manage the index manually.
            return state
            
        # This is a helper to handle the variable length of queries
        # But since we are in a reduce, we can't easily jump indices.
        # Instead, we will pre-group the queries.
        return state

    # Because the queries have variable lengths, a simple reduce over range(Q) 
    # requires a way to track the pointer in the input_data list.
    
    def query_parser(state, _):
        idx, current_time, plants, results = state
        q_type = input_data[idx]
        
        if q_type == '1':
            # Plant added at current_time
            # We use a list and maintain it sorted. 
            # Since we only append current_time and current_time is non-decreasing,
            # the list remains sorted.
            new_plants = plants + [current_time]
            return (idx + 1, current_time, new_plants, results)
        
        elif q_type == '2':
            t_val = int(input_data[idx + 1])
            return (idx + 1, current_time + t_val, plants, results)
            
        elif q_type == '3':
            h_val = int(input_data[idx + 1])
            # Harvest plants where birth_time <= current_time - h_val
            threshold = current_time - h_val
            # Find index of first plant with birth_time > threshold
            # All plants before this index are harvested.
            # Since we cannot mutate lists in reduce, we slice.
            # However, slicing creates a new list. To keep it efficient,
            # we use bisect_left on the sorted birth times.
            split_idx = bisect_left(plants, threshold + 1) 
            # Wait, the condition is height >= H, so birth_time <= current_time - H.
            # Correct threshold:
            threshold = current_time - h_val
            # We want count of plants where p <= threshold.
            # bisect_right finds the insertion point after any existing entries of threshold.
            import bisect
            count = bisect.bisect_right(plants, threshold)
            
            # The remaining plants are those from index 'count' onwards.
            return (idx + 2, current_time, plants[count:], results + [count])

    # To avoid the function definition inside and maintain purity for reduce:
    # We wrap the logic in a function and call it.
    
    # Since we cannot use loops, we use reduce over a range.
    # But the index jumps by 1 or 2. We handle this by passing the index in the state.
    # We use a range(Q*2) to ensure we have enough iterations, 
    # but we stop processing once the index reaches the end of input_data.
    
    # To handle the variable step, we can't use a simple range(Q).
    # Instead, we pre-process the input into a list of tuples.
    
    def group_queries(data):
        res = []
        i = 0
        while i < len(data):
            t = data[i]
            if t == '1':
                res.append((1, None))
                i += 1
            elif t == '2':
                res.append((2, int(data[i+1])))
                i += 2
            else:
                res.append((3, int(data[i+1])))
                i += 2
        return res

    queries = group_queries(input_data[1:])
    
    # State: (current_time, plants_list, results_list)
    def reducer(state, q):
        curr_time, plants, results = state
        q_type, val = q
        if q_type == 1:
            return (curr_time, plants + [curr_time], results)
        elif q_type == 2:
            return (curr_time + val, plants, results)
        else:
            # Height >= H  =>  curr_time - birth_time >= val  => birth_time <= curr_time - val
            import bisect
            threshold = curr_time - val
            idx = bisect.bisect_right(plants, threshold)
            return (curr_time, plants[idx:], results + [idx])

    final_state = reduce(reducer, queries, (0, [], []))
    
    sys.stdout.write('\n'.join(map(str, final_state[2])) + '\n')

if __name__ == "__main__":
    solve()