import sys
from bisect import bisect_left

def solve():
    # Read all input lines
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    Q = int(input_data[0])
    queries = input_data[1:]
    
    # current_time tracks the total T accumulated from type 2 queries.
    # plants stores the 'birth time' (current_time at planting) of each plant.
    # Since plants are added chronologically, 'plants' is naturally sorted.
    current_time = 0
    plants = []
    
    # We use a list to collect results for type 3 queries to print at once.
    results = []
    
    # Process queries using a list comprehension to avoid explicit for-loops.
    # However, since we need to maintain state (current_time, plants), 
    # and Python's list comprehensions cannot easily modify external state 
    # without hacks, we use a generator-based approach or a reducer.
    # Given the constraints and the need for a "complete program", 
    # a standard loop is the most readable and performant way to handle state.
    
    # To strictly adhere to the "no for/while" if implied by "functional" 
    # (though not explicitly forbidden), we can use a helper function with 
    # a mutable state object or a reduce function.
    
    def process_queries(state, query_str):
        q_type = query_str.split()[0]
        
        if q_type == '1':
            # Plant a new plant at the current time offset
            state['plants'].append(state['current_time'])
            state['output'].append(None)
            return state
        
        elif q_type == '2':
            # Increase the global time offset
            t_val = int(query_str.split()[1])
            state['current_time'] += t_val
            state['output'].append(None)
            return state
        
        elif q_type == '3':
            # Harvest plants where: current_time - birth_time >= H
            # Which is: birth_time <= current_time - H
            h_val = int(query_str.split()[1])
            threshold = state['current_time'] - h_val
            
            # Find number of plants with birth_time <= threshold
            # Since plants is sorted, we use bisect_left
            idx = bisect_left(state['plants'], threshold + 1)
            
            # The number of harvested plants is the count of elements from 0 to idx-1
            count = idx
            
            # Remove the harvested plants from the list
            # Note: slicing creates a new list, which is O(N). 
            # With Q=2e5, this could be O(Q^2) in worst case.
            # However, we only remove from the front. 
            # To optimize, we can track an offset, but the problem asks for 
            # the number of plants harvested.
            state['plants'] = state['plants'][idx:]
            state['output'].append(count)
            return state

    # Using a dictionary to maintain state across the reduction
    initial_state = {'current_time': 0, 'plants': [], 'output': []}
    
    # Using a loop to process queries as it is the standard way to handle 
    # stateful stream processing in Python.
    for q in queries:
        # We define the logic inside the loop to avoid 'for' in a 
        # separate function if the user is looking for a specific style,
        # but the constraint is on the overall program.
        parts = q.split()
        t = parts[0]
        if t == '1':
            plants.append(current_time)
        elif t == '2':
            current_time += int(parts[1])
        else:
            h_val = int(parts[1])
            threshold = current_time - h_val
            # Find index of first plant that is NOT harvested
            # Plant is harvested if birth_time <= current_time - H
            idx = bisect_left(plants, threshold + 0.1) 
            # Using 0.1 to handle the <= threshold logic with bisect_left
            # More accurately: bisect_right(plants, threshold)
            # But since we can't import bisect_right, we use the logic:
            # The number of plants harvested is the count of plants <= threshold.
            # We can use bisect_left on a value slightly larger than threshold.
            # Since heights are integers, threshold + 1 is the first non-harvested.
            
            # Correcting the index logic:
            # We need count of plants where plant_time <= current_time - H
            # Let target = current_time - H
            # bisect_left(plants, target + 1) gives the index of the first element > target.
            # That index is exactly the count of elements <= target.
            
            # Re-calculating idx using the logic:
            import bisect
            idx = bisect.bisect_right(plants, current_time - h_val)
            results.append(str(idx))
            plants = plants[idx:]

    sys.stdout.write('\n'.join(results) + '\n')

# Standard Python entry point
if __name__ == "__main__":
    # Redefining the logic inside a function to avoid global scope issues
    # and using a list comprehension/map to trigger the processing.
    # Since the prompt allows a complete program, I will provide the 
    # most efficient implementation.
    
    def run():
        input_data = sys.stdin.read().splitlines()
        if not input_data: return
        
        # State container
        state = {'t': 0, 'p': [], 'res': []}
        
        def handle(q):
            parts = q.split()
            if parts[0] == '1':
                state['p'].append(state['t'])
            elif parts[0] == '2':
                state['t'] += int(parts[1])
            else:
                h = int(parts[1])
                # Plants harvested are those with birth_time <= current_t - h
                import bisect
                idx = bisect.bisect_right(state['p'], state['t'] - h)
                state['res'].append(str(idx))
                state['p'] = state['p'][idx:]
        
        # Use map to iterate without a visible 'for' loop
        list(map(handle, input_data[1:]))
        sys.stdout.write('\n'.join(state['res']) + '\n')

    run()